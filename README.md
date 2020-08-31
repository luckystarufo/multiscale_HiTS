# Multiscale Hierarchical Time-Stepper (Multiscale HiTS)

### [Code](https://colab.research.google.com/drive/1I6sX-yqP__Z3iX-ita-pXi96d-tnZT_S?usp=sharing) | [Paper](https://arxiv.org/abs/2008.09768) | [Data](https://www.dropbox.com/sh/hn47hecp22xpxt4/AADkXmbqZHg4yPRnBAUMFi9wa?dl=0)

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
    * [Description](#description)
    * [Instruction](#instruction)
* [License](#license)
* [Citation](#citation)

## Introduction
In this work, we consider deep learning models in the context of scientific computing. 
We train neural networks (NNs) to perform time-stepping. The core idea of the proposed method is to 
couple NNs trained over various time scales together so that to formulate a 
<em>multiscale hierarchical time-stepper</em> (multiscale HiTS), as shown below.

![figure 1: method](./figures/Multiscale_forecast_diagram.jpeg?raw=true)

Our scheme provides important advantages:
* Highly accurate: time-steppers with small ∆t are responsible for the accurate time-stepping results over short periods, 
while the models with larger ∆t steps are used to ’reset’ the predictions over longer periods, preventing error accumulations 
from the short-time models.
* Highly efficient: the computation is easy to vectorize.
* Makes the training easier: each NN model only need to be trained over a short period, circumventing the 
[exploding/vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) problem.
* Super flexible: can be integrated with numerical time-steppers, resulting in hybrid time-steppers. The hybrid time-steppers
are parallelizable in nature, which is in sharp contrast to numerical time-stepping algorithms that are usually serialized.

Check the paper for more details.

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

Note 1: The folder names contained in <em>data/</em> and <em>models/</em> are the same: 
<em>Bach/</em>, <em>Cubic/</em>, <em>Flower/</em>, <em>Fluid/</em>, <em>Hopf/</em>, <em>Hyperbolic/</em>, <em>KS/</em>, 
<em>Linear/</em>, <em>Lorenz/</em>, <em>VanDerPol/</em>. 

Note 2: We don't upload the data, models and results to Github as they are large files. However, you will be able to 
generate them by following the instructions in the scripts and below.


## Getting started
We provide two ways for you to get started with this project. One is to use Google Colab and the other is to clone the 
repository and play with it locally.
### Colab
If you want to quickly experiment with HiTS, we have written a [Colab](https://colab.research.google.com/drive/1I6sX-yqP__Z3iX-ita-pXi96d-tnZT_S?usp=sharing). 
It outlines the big idea of our proposed multiscale HiTS and doesn't require installing anything or intensive training. 
Linear differential equation for a harmonic oscillator is served as a toy example to help you go through the core of the codes.
Figure 2 in the paper can be reproduced with it.

### Setup
However, for those who want to dig a bit more of the project, we suggest you to clone the project and run the experiments locally.
This work is mainly built on Python 3.7 and Pytorch. To set it up, we recommend you to create a virtual environment 
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

### Description
There are two key results shown in the paper. 
* The first one states that our multiscale HiTS outperforms any single
time-scale neural network time-steppers in terms of accuracy while also maintaining good efficiency thanks to the
vectorized computations of array programming. We also show that similar coupling technique can be applied to classic 
numerical time-steppers, resulting in hybrid time-steppers, accelerating classical numerical simulation algorithms. 
These results are illustrated by the following two figures in the paper.

![figure 2: exp11](./figures/multiscale_forecast.jpeg?raw=true)

![figure 3: exp12](./figures/acc_vs_eff.jpeg?raw=true)

* The second result shows multiscale HiTS outperforms state-of-the-art RNN architectures, including LSTM, ESN and CW-RNN,
over the sequence generation task, which is shown by the figure below. There's also a [video demo](https://www.youtube.com/watch?v=2psX5efLhCE) 
for it. The sequences we explore include a simulated solution of the Kuramoto–Sivashinsky (KS) equation, 
a music excerpt from Bach’s Fugue No. 1 In C Major, BWV 846, a simulation of fluid flow past a circular cylinder at 
Reynolds number 100, and a video frame of blooming flowers, which you can find [here](https://www.dropbox.com/sh/hn47hecp22xpxt4/AADkXmbqZHg4yPRnBAUMFi9wa?dl=0).

![figure 4: exp21](./figures/benchmarks.jpeg?raw=true)

### Instruction
All results can be reproduced with the help of the notebooks in <em>scripts/</em>, though you may need a GPU machine for
training some of the neural networks. 
* For the first set of experiments, please use the scripts in <em>multiscale_HiTS_exp/</em>. You should first run 
<em>data_generation.ipynb</em> to generate the data sets then train the neural network time-steppers with <em>model_training.ipynb</em>.
After that, you can run the other three scripts to reproduce Table 5 - 9, Figure 5, 6, 8, 9.
* For the second set of experiments, please refer to <em>sequence_generation_exp/</em>. You will be able to get Table 1 
and Figure 7 with the setup and training details documented in the Appendix of the paper. 
* There is also a <em>others/</em> in the <em>scripts/</em> folder. The two notebooks in it can be used to generate Figure 2 
and 10. Note that one of the two notebooks, which is named <em>motivating_example.ipynb</em>, is pretty much the same with 
the one provided in [Google Colab](https://colab.research.google.com/drive/1I6sX-yqP__Z3iX-ita-pXi96d-tnZT_S?usp=sharing).

Happy coding :)


## License
This project utilizes the [MIT LICENSE](LICENSE).
100% open-source, feel free to utilize the code however you like. 

## Citation
```
@article{liu2020hierarchical,
  title={Hierarchical Deep Learning of Multiscale Differential Equation Time-Steppers},
  author={Liu, Yuying and Kutz, J Nathan and Brunton, Steven L},
  journal={arXiv preprint arXiv:2008.09768},
  year={2020}
}
```










