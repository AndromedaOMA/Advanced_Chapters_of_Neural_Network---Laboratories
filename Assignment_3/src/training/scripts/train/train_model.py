from Assignment_3.src.data_pipeline.preprocessing.preprocessing import preprocessing
from Assignment_3.src.training.utils.get_loss_function import get_loss_function
from Assignment_3.src.training.utils.get_lr_scheduler import get_lr_scheduler
from Assignment_3.src.training.utils.get_model import get_model
from Assignment_3.src.training.utils.get_optimizer import get_optimizer
from Assignment_3.src.training.utils.load_config import load_config

if __name__ == '__main__':
    config = load_config("../../experiments/experiment1/config.yml")

    print(f'Running the {config["experiment"]["name"]} experiment')

    print(f'Device - {config["experiment"]["device"]}')
    device = config['experiment']['device']

    train_loader, test_loader = preprocessing(config)

    model = get_model(config['model']['name'])
    loss_function = get_loss_function(config["training"]["loss_function"])
    optimizer = get_optimizer(config, model.parameters())
    train_scheduler = get_lr_scheduler(config, optimizer)
    # Batch size scheduler


