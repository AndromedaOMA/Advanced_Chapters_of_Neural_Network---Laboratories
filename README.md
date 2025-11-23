<h1 align="center">Hi 👋, here we have the Advanced_Chapters_of_Neural_Network Laboratories</h1>
<!-- <h3 align="center"> </h3> -->


## Table Of Content
* [Assignment_3: Developed a PyTorch Training Pipeline](#assignment-3)
* [Setup](#setup)
* [How_to_run](#htr)

--------------------------------------------------------------------------------

<h1 id="assignment-3" align="left">Assignment_3: PyTorch Training Pipeline</h1>

<h3 align="left">Here we have the requirement:</h3>

Implement a generic training pipeline using PyTorch. This pipeline must be configurable either through command line arguments or configuration files. Prepare
a short report with the experimental results. The report can be Markdown or PDF file, should have at least 1 page, and will be included in the submission.

<h3 align="left">Specifications:</h3>

1. Pipeline is device agnostic.
2. You can configure classification training on the following datasets: MNIST, CIFAR-10, CIFAR-100, and OxfordIIITPet.
3. Datasets are efficient and support data augmentation.
4. Can use any of the following models: resnet18, resnet50, resnest14d, resnest26d, and MLP. Use timm or huggingface to load the models.
5. Can be configured to use any of the following optimizers: SGD, Adam,
AdamW, Muon, SAM.
6. Can be configured to use one of the following learning rate schedulers: StepLR and ReduceLROnPlateau.
7. Integrates a batch size scheduler.
8. Is integrated with Tensorboard and/or wandb for metrics reporting. Must report relevant training and testing metrics. Supports an early stopping mechanism.

• Do a hyperparameter sweep using Wandb or a custom script + Tensorboard.

• Have at least 8 configurations that achieve over 70% accuracy on the CIFAR-100 dataset.

• Describe the parameters you varied during the sweep in the report.

• You must mention that the Jax configuration is faster. Add a table with the test accuracy for your experiments and the time spent for training.

• Include pictures from your metrics reporting system (Tensorboard or wandb).

• The training pipeline is efficient (training time or RAM usage or VRAM usage).

• You motivate why in the report and include measurements.

• You must mention that the Jax configuration is faster. Caching will not be considered, you have to do several other steps to achieve this.


<h3 align="left">How does it work?</h3>

  
<h3 align="left">The logic behind the code:</h3>

The pipeline is based on a reusable and reconfigurable training module. Each experiment is associated with a configuration file based on which the chosen model will be trained. The configurations contain the hyperparameters of the entire pipeline!

Below you will find a structure of the project files which also provides additional explanations:

Assignment_3/
│
├── data/                          # All data-related directories and files
│   ├── MNIST/                     # Original, immutable data dump
│   ├── CIFAR10/                   # Original, immutable data dump
│   ├── CIFAR100/                  # Original, immutable data dump
│   └── OxfordIIITPet/             # Original, immutable data dump
│
├── docs/                          # Project documentation
│   ├── environment.yaml           # Anaconda environment required to setup
│   ├── requirements.txt           # Python package requirements file
│   └── file_structure.txt         # Project files structure
│
├── src/                           # Source code
│   ├── data_pipeline/             # All modules related to the data pipeline
│   │   └── preprocessing/         # Data preprocessing modules
│   │       └── preprocessing.py   # Data preprocessing module
│   │
│   ├── models/                    # Neural network models and components
│   │   ├── architectures/         # Different neural network architectures
│   │   └── layers/                # Custom layers
│   │
│   └── training/
│       ├── experiments/           # Specific experiment settings and results
│       │   ├── experiment1/       # Each experiment can have its own subdirectory
│       │   │   ├── config.yml     # Configuration file for the experiment
│       │   │   └── results/       # Resulting logs via tensorboard
│       │   ├── experiment2/
│       │   └── ...
│       │
│       ├── scripts/               # Actual scripts to run training
│       │   ├── wandb              # Wandb logs
│       │   └── train_model.py     # Main training script
│       │
│       └── utils/                 # Miscellaneous utilities for training
│           ├── weight_initialization.py # Weight initialization strategies
│           ├── load_config.py     # Config loader
│           ├── get_loss_function.py # Loss Function Getter
│           ├── get_lr_scheduler.py # Learning Rate Scheduler Getter
│           ├── get_model.py       # Model Getter
│           ├── get_optimizer.py   # Optimizer Getter
│           └── mixed_precision.py # Mixed precision training utilities
│
└── README.md                      # Overview and instructions for the project
  
---

<h3 id="score" align="left">Best score:</h3>

<img src="https://github.com/user-attachments/assets/1b53b7f2-bc87-4ee5-ae97-14bdf6a11f06" alt="Moments before the disaster" style="width: 300px; height: auto;">

---

<h3 id="setup" align="left">Setup:</h3>

1. Clone the current repositoy! Now you have the project avalable on your local server!</br>
 Type command: ```git clone git@github.com:AndromedaOMA/Neural_Networks---Laboratories.git```
2. Select, open and run the chosen project through PyCharm IDE or the preferred IDE.
3. Have fun!

---

<h3 id="htr" align="left">How to run:</h3>

1. Clone the current repositoy! Now you have the project avalable on your local server!</br>
 Type command: ```git clone git@github.com:AndromedaOMA/Neural_Networks---Laboratories.git```
2. Select, open and run the chosen project through PyCharm IDE or the preferred IDE.
3. Have fun!

---

- ⚡ Fun fact: **tba!**

* [Table Of Content](#table-of-content)

