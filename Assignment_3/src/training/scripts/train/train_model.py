from Assignment_3.src.data_pipeline.preprocessing.preprocessing import preprocessing
from Assignment_3.src.training.utils.load_config import load_config

if __name__ == '__main__':
    config = load_config("../../experiments/experiment1/config.yml")

    print(f'Running the {config['experiment']['name']} experiment')
    device = config['experiment']['device']

    print(f'Device - {config['experiment']['device']}')
    train_loader, test_loader = preprocessing(config)
