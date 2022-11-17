# COM 3031 Computational Intelligence Coursework 

## Memeber of a groups

1. Wish Suharitdamrong

2. Taimoor Rizwan

3. Ionut Boston

## Installation

Use Anaconda virtual environment to install all dependencies

```bash
conda create -n ci # create virtual environment
conda activate ci # activate virtual environment
```
### Use anaconda to install Pytorch

#### For MacOs (only CPU)
```bash
conda install pytorch torchvision torchaudio -c pytorch
```
#### For Linux (Both GPU and CPU)
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

#### Use pip to install other packages

```bash
pip install -r requirement.txt
```


## Run

```bash
python train_gd.py #train classifier
```

## Visualization

```bash
Tensordboard --logdir=../CI_logs
```

