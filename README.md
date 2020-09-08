# Parameter-Learning-In-Tumor-Environments
This work provides a data driven approach to learn the key parameters that are guiding the spacial organisation of the tumor environment. A method to learn a parameter set of manageable size from pathological images is presented. This set of connectivity parameters is used as input for a generative model. The goal is to generate a pattern configuration from a simulation, which represents the macroscopic key features of the original pathological image, to identify key principles of the cellular organisation and to quantify the interaction strength that is responsible for the resulting organisation.

Keywords: Tumor Biology, Cellular Potts Model, Parameter Inference, Maximum Pseudo Likelihood, Monte Carlo Optimisation

### Work flow: 
1.  Load cellData 
2.  Set parameters for: maximum pseudo likelihood parameter inference and generating MCMC simulation
3.  Infere parameters from image with MPL: paramInference()
4.  Simulate with infered connectivity matrix: mcGenerator()


### Getting started

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import methods1lib as lb1
import mclib as mclb
import mpllib as mplb

#%% Load data
cell_data = pd.read_csv("cellData.csv",delimiter= ",")
cell_data = cell_data.to_numpy(); cell_data = cell_data.astype(float) 

cellTypes_Detailed = np.array(['Undefined','Endothelial','Mesenchymal-like','Tumor & K+','Treg','CD4_T',
                      'CD8-T','CD3-T','NK','B','Neutrophils','Macrophages & Mono/Neu','DC & DC/Mono','Other Immune'])
cellTypes_Grained = np.array(['Other','Endothelial','Mesenchymal-like','Tumor & K+','T-cells','NK','B',
                     'Granulocytes','DCs'])
cellTypes_General = np.array(['Other','Immune','Endothelial','Mesenchymal-like','Tumor'])

#%% Set Parameters
#--Simulation specific Parameters
patient = 4
depiction = 'General' #'Detailed'; 'Grained'
cellTypes_orig = [len(cellTypes_General)]
limiter = False
runs = 1 #number of optimisations for every sample

#--MCMC Encoding Parameters
N = 150 #MC Epochs
boxlength = 780. #micro meter
TT_enc = np.array([1.]); Tsteps = len(TT_enc) #MC Temperature
#--MPL Decoding Parameters
TT_dec = TT_enc #MPL decoding temperature
```

### Cell Type Depictions
 
  General | Grained | Detailed  
:-------------------------:|:-------------------------:|:-------------------------:
 <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A1decay.png" width="320" height="300" />  |  <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A1decay.png" width="320" height="300" /> |  <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A2decay.png" width="320" height="300" />  
 
Projection of the high dimensional gene- expression tumorsamples onto the first two principle components. The three vertexpoints/ archetypes highlighted correspond to distinct features andspan the triangle of the pareto front, such that all points within canbe explained as convex combinations of the vertices.

### Biological Interpretation
  A0 | A1 | A2  
:-------------------------:|:-------------------------:|:-------------------------:
 <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A1decay.png" width="320" height="300" />  |  <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A1decay.png" width="320" height="300" /> |  <img src="https://github.com/david-alber/Infearing-Key-Biological-Tasks/blob/master/Images/A2decay.png" width="320" height="300" />

Selection of one GO- expression for each archetype, according to the maximal descent away from the archetype, i.e. most negative slope for linear regression.  Suggesting that Biocarta Blymphocyte Phathway, Reactome Unwinding of DNA and Reactome Endosomal Vacuolary Pathwaycan be linked to key biological tasks.

### How to contribute
Fork from the `Developer`- branch and pull request to merge back into the original `Developer`- branch. 
Working updates and improvements will then be merged into the `Master` branch, which will always contain the latest working version.

With: 
* [Lukas Alber](https://github.com/luksen99)
* [Jean Hausser](https://www.scilifelab.se/researchers/jean-hausser/)

### Dependencies
 [py_pchy](https://pypi.org/project/py-pcha/), 
 [scikit-learn](https://scikit-learn.org/stable/), 
 [Numpy](https://numpy.org/), 
 [Scipy](https://www.scipy.org/), 
 [Matplotlib](https://matplotlib.org/), 
 [Pandas](https://pandas.pydata.org/)
 
 
 [1]:https://arxiv.org/abs/1901.10799
 [2]:https://www.tandfonline.com/doi/abs/10.1080/00401706.1994.10485840
 [3]:https://science.sciencemag.org/content/336/6085/1157/tab-article-info
 [4]:https://www.nature.com/articles/nmeth.3254
 
