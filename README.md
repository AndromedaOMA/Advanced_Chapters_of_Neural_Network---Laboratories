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
5. Can be configured to use any of the following optimizers: SGD, Adam, AdamW, Muon, SAM.
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

• The training pipeline is efficient (training time or RAM usage or VRAM usage).

• You motivate why in the report and include measurements.

  
<h3 align="left">The logic behind the code:</h3>

The pipeline is based on a reusable and reconfigurable training module. Each experiment is associated with a configuration file based on which the chosen model will be trained. The configurations contain the hyperparameters of the entire pipeline!

Below you will find a structure of the project files which also provides additional explanations:
### Project Structure

```
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
│           ├── weight_initialization.py   # Weight initialization strategies
│           ├── load_config.py             # Config loader
│           ├── get_loss_function.py       # Loss Function Getter
│           ├── get_lr_scheduler.py        # Learning Rate Scheduler Getter
│           ├── get_model.py               # Model Getter
│           ├── get_optimizer.py           # Optimizer Getter
│           └── mixed_precision.py         # Mixed precision training utilities
│
└── README.md                      # Overview and instructions for the project
```

---

<h3 id="WB-chart" align="left">W&B Chart:</h3>

<!-- Main large chart -->
<div style="margin-bottom:15px;">
  <img 
    src="https://github.com/user-attachments/assets/a0dd52cf-667b-4d43-8f0e-33c45e4556df"
    width="80%"
    style="border-radius:10px; box-shadow:0 0 8px rgba(0,0,0,0.25);"
  />
</div>

<!-- Three small charts (left-aligned) -->
<div style="display:flex; gap:20px; justify-content:flex-start;">
  <img src="https://github.com/user-attachments/assets/01be89a5-2a3f-46af-8765-54a0f52f0723"
       width="250"
       style="border-radius:10px; box-shadow:0 0 8px rgba(0,0,0,0.25);" />
  <img src="https://github.com/user-attachments/assets/f40867f9-eafa-4d04-a88e-83699a52d090"
       width="250"
       style="border-radius:10px; box-shadow:0 0 8px rgba(0,0,0,0.25);" />
  <img src="https://github.com/user-attachments/assets/ad8c0fee-27d7-46c1-96ba-5909b403eae2"
       width="250"
       style="border-radius:10px; box-shadow:0 0 8px rgba(0,0,0,0.25);" />
</div>


<h3 id="table" align="left">Table of hyperparameters:</h3>

| val_acc | Runtime | batch | optim | train_loss | val_loss |
|--------:|--------:|-------:|:------|-----------:|----------:|
| 76.47 | 350 | 128 | AdamW | 0.17224 | 1.05642 |
| 79.29 | 1169 | 64 | AdamW | 0.0449337 | 1.00261 |
| 78.87 | 1110 | 128 | AdamW | 0.0681302 | 0.965091 |
| 78.53 | 1654 | 32 | AdamW | 0.412467 | 1.03267 |
| 77.25 | 1336 | 128 | SGD | 0.289078 | 0.978043 |
| 79.45 | 1352 | 128 | AdamW | 0.0103327 | 0.961207 |
| 77.38 | 1195 | 128 | SGD | 0.143339 | 1.00924 |
| 79.05 | 1094 | 128 | AdamW | 0.0238482 | 0.962663 |
| 79.01 | 1131 | 64 | AdamW | 0.155995 | 1.05255 |
| 79.02 | 1136 | 64 | AdamW | 0.247853 | 0.975867 |
| 76.47 | 1074 | 128 | SGD | 1.01373 | 1.2991 |
| 78.35 | 1426 | 32 | AdamW | 0.144093 | 1.01578 |
| 78.5 | 1143 | 64 | Adam | 0.342168 | 0.961861 |
| 78.52 | 1116 | 128 | Adam | 0.0861193 | 0.965395 |
| 71.64 | 470 | 32 | AdamW | 1.41954 | 1.1866 |
| 76.72 | 185 | 64 | SGD | 0.329836 | 0.989844 |
| 75.76 | 362 | 64 | SGD | 2.01752 | 1.85608 |
| 77.64 | 367 | 128 | SGD | 0.245117 | 0.926565 |
| 77.35 | 341 | 128 | SGD | 0.108086 | 0.968217 |

The four hyperparameters that were varied during the sweep are (as you can see above):

1. Batch Size

2. Learning Rate

3. Optimizer (SGD, Adam, AdamW)

4. Weight Decay (for Adam/AdamW)

---

<h3 align="left">How does is this pipeline efficient?</h3>

```scaler = get_mixed_precision()``` typically returns a mixed-precision gradient scaler, usually a wrapper around torch.cuda.amp.GradScaler (in PyTorch) or an equivalent object in other frameworks.

Purpose:

1. Enables mixed-precision training (FP16 + FP32) to **speed up training** on GPUs with Tensor Cores.

2. Prevents numerical underflow by scaling the loss before backpropagation.

3. Works with autocast() to automate mixed-precision operations.

---

<h3 id="setup" align="left">Setup:</h3>

1. Clone the current repositoy! Now you have the project avalable on your local server!</br>
 Type command: ```git clone git@github.com:AndromedaOMA/Neural_Networks---Laboratories.git```
2. Select, open and run the chosen project through PyCharm IDE or the preferred IDE.
3. Set the Anaconda environment using ```Assignment_3/docs/environment.yaml``` requirements/configuration file.
4. Have fun!

---

<h3 id="htr" align="left">How to run:</h3>

1. Go to ```Assignment_3/src/training/experiments``` and choose one experiment sub directory.
2. Each experiment sub directory contains a ```config.yml``` file. Here you can change easly the hyperparameters and the dataset you want to train the model you want.
3. Running the ```Assignment_3/src/training/scripts/train_model.py``` script will call a hyperparameter sweep based on your modiffied config.yml file.

---

- ⚡ Fun fact: **tba!**

* [Table Of Content](#table-of-content)











