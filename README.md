# COM 3031 Computational Intelligence Coursework 

## Memeber of a groups

1. Wish Suharitdamrong

2. Taimoor Rizwan

3. Ionut Boston

## Installation

There are two ways of installing package using conda

1.Create virtual conda environment from ```environment.yml ```

2.Use conda and  pip to install a pakages


### 1.Create Virtual Environment from environment.yml

```bash
# Create virtual environment from .yml file
conda env create -f environment.yml

# activate virtual environment
conda activate ci 
```

### 2.Create Virtual Environment

Use Anaconda virtual environment to install all dependencies

```bash
# create virtual environment
conda create -n ci 

# activate virtual environment
conda activate ci 
```
#### Use anaconda to install Pytorch

##### For MacOs (only CPU)
```bash
conda install pytorch torchvision torchaudio -c pytorch
```
##### For Linux (Both GPU and CPU)
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

##### Use pip to install other packages

```bash
pip install -r requirement.txt
```

## Directory

##### Pretrain weight directory
```bash
├── ckpt # checkpoint of a pretrain model
│   └── CIFAR-10_GD_SGD.pth
```

## Run

### Simple training

```bash
python train_gd.py # Use gradient descent to train whole network
python train_ga.py # Use genetic algorithms to train whole network
python train_pso.py # Use particle swarm optimization to train whole network
```


### Two stage training 



```bash
python train_hybrid_*.py # All the fiels start with train_hybrid are two stage training
python train_meme.py # Train two stage with memetic algorithms
python train_nsga.py # Train two stage with NSGA II algorithms
```

## Visualization

#### Visualize result in jupyter notebook
```bash
jupyter notebook plot.ipynb
```

#### Visualization the results in Tensorboard
```bash
Tensordboard --logdir=../CI_logs 
```
