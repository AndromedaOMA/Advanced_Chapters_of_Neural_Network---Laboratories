import os
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from Assignment_3.src.data_pipeline.preprocessing.preprocessing import preprocessing
from Assignment_3.src.training.utils.get_loss_function import get_loss_function
from Assignment_3.src.training.utils.get_lr_scheduler import get_lr_scheduler
from Assignment_3.src.training.utils.get_model import get_model
from Assignment_3.src.training.utils.get_optimizer import get_optimizer
from Assignment_3.src.training.utils.load_config import load_config
from Assignment_3.src.training.utils.mixed_precision import get_mixed_precision


scaler = get_mixed_precision()


def train(epoch):
    model.train()
    train_loss = 0.0

    for batch_index, (train_images, train_labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        train_images, train_labels = train_images.to(device), train_labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device):  # mixed precision
            outputs = model(train_images)
            loss = loss_function(outputs, train_labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        n_iter = (epoch - 1) * len(train_loader) + batch_index + 1
        writer.add_scalar("Train/Loss", loss.item(), n_iter)

        # Log gradients of the last layer
        last_layer = list(model.children())[-1]
        for name, param in last_layer.named_parameters():
            if param.grad is not None:
                writer.add_scalar(f"LastLayerGradients/{name}", param.grad.norm(), n_iter)

        train_loss += loss.item()

    if config['training']['scheduler'] == 'stepLR':
        scheduler.step()
    else:
        scheduler.step(train_loss)

    # Log parameter histograms per epoch
    for name, param in model.named_parameters():
        writer.add_histogram(name.replace('.', '/'), param, epoch)

    return train_loss / len(train_loader)


@torch.no_grad()
def eval_training(epoch=0):
    model.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0

    for (test_images, test_labels) in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)

        outputs = model(test_images)
        test_loss += loss_function(outputs, test_labels).item()

        prediction = outputs.argmax(dim=1)
        correct += (prediction == test_labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f"Epoch {epoch}: Test Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}%")

    if writer:
        writer.add_scalar("Test/Loss", avg_loss, epoch)
        writer.add_scalar("Test/Accuracy", accuracy, epoch)

    return accuracy


if __name__ == '__main__':
    print("Type the number of the experiment you want to run:")
    experiment_number = int(input())
    config = load_config(f"../experiments/experiment{experiment_number}/config.yml")
    print(f'Running the experiment{experiment_number}')

    device = config['experiment']['device']
    print(f'Running on {device} device')

    train_loader, test_loader = preprocessing(config)

    model = get_model(config['model']['name'], config).to(device)
    loss_function = get_loss_function(config["training"]["loss_function"])
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_lr_scheduler(config, optimizer)
    # Batch size scheduler

    log_dir = f'../experiments/experiment{config["experiment"]["number"]}/results'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    writer = SummaryWriter(
        log_dir=os.path.join(
            log_dir, datetime.now().strftime(config["experiment"]["date_format"])
        )
    )

    sample_images, _ = next(iter(train_loader))
    writer.add_graph(model, sample_images[:1].to(device))

    best_acc = 0.0
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss = train(epoch)
        acc = eval_training(epoch)

        if acc > best_acc:
            best_acc = acc
            checkpoint_dir = f'../experiments/experiment{config["experiment"]["number"]}/checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'best_model_{best_acc}_acc.pth'))

    writer.close()
