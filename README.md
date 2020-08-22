# Multiscale Hierarchical Time-Stepper (Multiscale HiTS)

### [Code](https://colab.research.google.com/drive/1I6sX-yqP__Z3iX-ita-pXi96d-tnZT_S) | [Paper]() | [Data](https://www.dropbox.com/sh/hn47hecp22xpxt4/AADkXmbqZHg4yPRnBAUMFi9wa?dl=0)

[Yuying Liu](https://students.washington.edu/yliu814/wordpress/),
[J. Nathan Kutz](http://faculty.washington.edu/kutz/),
[Steven L. Brunton](https://www.eigensteve.com), University of Washington. <br>

The purpose of this repository is to help users reproduce the results shown in the paper "Hierarchical Deep Learning of Multiscale Differential Equation Time-Steppers".

## Table of contents
* [Introduction](#introduction)
* [Structure](#structure)
* [Getting Started](#getting-started)
    * [Colab](#colab)
    * [Setup](#setup)
* [Results](#results)
* [License](#license)
* [Citation](#citation)

## Introduction

## Structure
    multiscale_HiTS/
        |- data/
            |- Linear/
            |- Hyperbolic/
            |- ...
            |- KS/
        |- src/
            |- ResNet.py
            |- utils.py
            |- rnn.py
            |- esn.py
            |- cwrnn.py
        |- scripts/
            |- multiscale_HiTS_exp/
                |- data_generation.ipynb
                |- model_training.ipynb
                |- ...
                |- multiscale_HiTS.ipynb
            |- sequence_generation_exp/
                |- Bach.ipynb
                |- ...
                |- seq_generations.ipynb
            |- others/
                |- motivating_example.ipynb
                |- visualize_increment.ipynb
        |- models/
            |- Linear/
            |- Hyperbolic/
            |- ...
            |- KS/
        |- results/
            |- Bach/
            |- Flower/
            |- Fluid/
            |- KS/
        |- figures/
        |- requirements.txt
        |- README.md
        |- LICENSE


## Getting started
We provide two ways for you to get started with this project. One is to use Google Colab and the other is to clone the 
repository and play with it locally.
### Colab
If you want to quickly experiment with HiTS, we have written a [Colab](https://colab.research.google.com/drive/1I6sX-yqP__Z3iX-ita-pXi96d-tnZT_S). 
It outlines the big idea of our proposed multiscale HiTS and doesn't require installing anything or intensive training. 
Linear differential equation for a harmonic oscillator is served as a toy example to help you go through the core of the codes.
Figure 2 in the paper can be reproduced with it.

### Setup
However, for those who want to dig a bit more of the project, we suggest you to clone the project and run the experiments locally.
This work is mainly built on python 3.7 and Pytorch. To set it up, I recommend you to create a virtual environment 
using [Anaconda](https://docs.anaconda.com/anaconda/install/). Once you get it correctly installed, run
```
git clone https://github.com/luckystarufo/multiscale_HiTS.git
conda create -n <ENV_NAME> python=3.7
conda activate <ENV_NAME>
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```
to restore the environment we use. If you are a Jupyter notebook user, run the following inside the environment:
```
pip install --user ipykernel
python -m ipykernel install --user --name=<ENV_NAME>
```
To allow tqdm (the progress bar library) to run in a notebook, you also need:
```
conda install -c conda-forge ipywidgets
```
Almost all codes are put in Jupyter Notebook (.ipynb) files to encourage exploration and modification of the work. 


## Results


## License
This project utilizes the [MIT LICENSE](LICENSE).
100% open-source, feel free to utilize the code however you like. 

## Citation
```
TBD
```










