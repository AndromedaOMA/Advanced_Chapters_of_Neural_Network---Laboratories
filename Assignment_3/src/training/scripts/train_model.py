import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from Assignment_3.src.data_pipeline.preprocessing.preprocessing import preprocessing
from Assignment_3.src.training.utils.get_loss_function import get_loss_function
from Assignment_3.src.training.utils.get_lr_scheduler import get_lr_scheduler
from Assignment_3.src.training.utils.get_model import get_model
from Assignment_3.src.training.utils.get_optimizer import get_optimizer
from Assignment_3.src.training.utils.load_config import load_config


if __name__ == '__main__':
    config = load_config("../experiments/experiment1/config.yml")

    print(f'Running the {config["experiment"]["name"]} experiment')

    print(f'Device - {config["experiment"]["device"]}')
    device = config['experiment']['device']

    train_loader, test_loader = preprocessing(config)

    model = get_model(config['model']['name']).to(device)
    loss_function = get_loss_function(config["training"]["loss_function"])
    optimizer = get_optimizer(config, model.parameters())
    train_scheduler = get_lr_scheduler(config, optimizer)
    # Batch size scheduler

    log_dir = f'../experiments/experiment{config["experiment"]["number"]}/results'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    writer = SummaryWriter(
        log_dir=os.path.join(
            log_dir, datetime.now().strftime(config["experiment"]["date_format"])
        )
    )
    images, labels = next(iter(train_loader))
    b, c, w, h = images.shape
    input_tensor = torch.Tensor(1, c, w, h).to(device)
    writer.add_graph(model, input_tensor)
