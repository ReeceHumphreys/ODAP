<p align="center"><a href="https://github.com/reecehumphreys/ODAP"><img src="https://github.com/reecehumphreys/ODAP/blob/master/images/ODAP.png" alt="Planet header image" height="150"/></a></p>

<h1 align="center">ODAP</h1>
<p align="center">Orbital Debris Analysis with Python.</p>

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
