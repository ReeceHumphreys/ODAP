<p align="center"><a href="https://github.com/reecehumphreys/ODAP"><img src="https://github.com/reecehumphreys/ODAP/blob/master/images/ODAP.png" alt="Planet header image" height="150"/></a></p>

<h1 align="center">ODAP</h1>
<p align="center">Orbital Debris Analysis with Python.</p>

<p align="center">
  <img src="https://img.shields.io/lgtm/grade/python/github/ReeceHumphreys/ODAP?style=flat-square" />
  <a href="https://github.com/ReeceHumphreys/odap/blob/development/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/github/license/ReeceHumphreys/ODAP?style=flat-square" target="_blank" />
  </a>
  <a href="https://twitter.com/ReeceWHumphreys">
    <img alt="Twitter: ReeceWHumphreys" src="https://img.shields.io/twitter/follow/ReeceWHumphreys.svg?style=social" target="_blank" />
  </a>
</p>

## Contents
- [Introduction](#introduction)
- [Progress](#overhaul-progress)
- [Setup](#setup-environment)

## Introduction

Orbital Debris Analysis with Python (ODAP) is an in-progress python package that implements the NASA Standard Breakup Model for simulating fragmentation events. It also includes tools for further orbital debris analysis, such as an orbit propagator that includes perturbations and various visualization tools. 


## Overhaul Progress
#### ODAP is currently undergoing a major overhaul to convert it into an open python package as such, functionality is influx and some features may be broken. Below shows the current progress torward this overhaul.

- [x] Refactor NASA Breakup model implementation
- [ ] Overhaul Jupyter notebook
- [ ] Implement API docs
- [ ] Add Testing + full code coverage  
- [ ] Refactor orbit propagator
- [ ] Refactor visualization tools
- [ ] Implement addtional analysis tools


## Setup Environment

Install conda:
https://www.anaconda.com/products/individual#Downloads

In base environment install jupyter:
```
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels // Allows us to use env kernels in base env
```

(Optional) Install Jupyter Extensions 
```
conda install -c conda-forge jupyter_contrib_nbextensions
```

Install pip in base environment
```
conda install pip
```

Create ODAP conda environment to contain dependencies:
```
conda create -n odap-env python=3.9
```

Activate odap-env environment:
```
conda activate odap-env
```

Install depencies inside of ODAP Conda env:
```
(odap-env) $ conda install pip
...
```

Install ipykernel inside of ODAP Conda env:
```
(odap-env) $ conda install ipykernel
```

Switch back to base environment:
```
(odap-env) $ conda deactivate
```

Activate Jupyter from base Conda env:
```
jupyter notebook
```

Select ODAP Kernel from Jupyter
(Should appear something like Python [conda env:odap-env])
