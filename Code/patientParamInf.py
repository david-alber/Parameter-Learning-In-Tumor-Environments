# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:31:13 2020

@author: alber
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import methods1lib as lb1
import mclib as mclb
import mpllib as mplb
plt.close('all')

#Dependencies: methods1lib(custom); mclib(custom); mplib(custom); 
#              pandas; matplotlib; numba; scipy; numpy; tqdm; scikit-image

#Workflow:
    #Load cell_center_data (from comAna.py)
    #Set parameters for: MPL param inference; MCMC simulation
    #Infere parameters from image with MPL: paramInference()
    #Simulate with infered connectivity matrix: mcGenerator()

#%%---Input image and parameters
#cell_center data with immune group tag. [n_cells with zeros,x/y/type,patient]
#Comments: cell_center_data_grain organisation: patient:p4_Detailed/ p4_Grained/ p4_General
#                                                       p12_Detailed/ p12_Grained/ p12_General
cell_data = pd.read_csv("cellData.csv",delimiter= ",")
cell_data = cell_data.to_numpy(); cell_data = cell_data.astype(float)        
cellTypes_Detailed = np.array(['Undefined','Endothelial','Mesenchymal-like','Tumor & K+','Treg','CD4_T',
                      'CD8-T','CD3-T','NK','B','Neutrophils','Macrophages & Mono/Neu','DC & DC/Mono','Other Immune'])
cellTypes_Grained = np.array(['Other','Endothelial','Mesenchymal-like','Tumor & K+','T-cells','NK','B',
                     'Granulocytes','DCs'])
cellTypes_General = np.array(['Other','Immune','Endothelial','Mesenchymal-like','Tumor'])

#%%--PARAMETERS
#--Simulation specific Parameters
patient = 4
depiction = 'Grained'
cellTypes_orig = [len(cellTypes_Grained)]
limiter = False
runs = 1 #number of optimisations for every sample

#--MCMC Encoding Parameters
N = 150 #MC Epochs
boxlength = 780. #micro meter
TT_enc = np.array([1.]); Tsteps = len(TT_enc) #MC Temperature
#--MPL Decoding Parameters
TT_dec = TT_enc #MPL decoding temperature

#%%---Reading cell config from data
print(f'\n Tissue image -> center of mass; Patient: {patient}; Depiction '+depiction)
cell_center_data = np.zeros((9000,3,1))
cell_centers, immuneGroup, cellTypes  = lb1.pathology(cell_data,patient,depiction=depiction, #ImmuneGroup #CellGroup
                                                      plotArr=True,plotImg=True)
#store cell_center data with immune group tag. [n_cells with zeros,x/y/type,patient]
cell_center_data[0:len(immuneGroup),:2,0] = cell_centers
cell_center_data[0:len(immuneGroup),2,0] = immuneGroup

#%%---MPL Inference: DECODE Image
Jmat_optArr,evalvec,rlimit,avg_neighb,n_cells = mplb.paramInference(cell_center_data,
                                                                    runs,limiter,TT_dec,boxlength) 
#%% Save
np.save("Jmat_opt_%s_%d" %(depiction,patient),Jmat_optArr)  #Infered parameters: [optRow,optCol,runs,sample]
np.save("evalvec_%s_%d" %(depiction,patient),evalvec) #mpl value for infered parameters: [runs, sample]
np.save("rlimit_%s_%d" %(depiction,patient),rlimit) #limiting radius: [sample]
np.save("avg_neighb_%s_%d" %(depiction,patient),avg_neighb) #average number of neighbors: [sample]
np.save("n_cells_%s_%d" %(depiction,patient),n_cells) #nuber of cells for each patient: [sample]

#%%---MCMC Evaluation: ENCODE Connectivity
types_inf,Energy_vec,T,mcStep = mclb.mcGenerator(Jmat_optArr,cell_center_data,rlimit,
                                                 runs,N,limiter,TT_enc,boxlength,cellTypes_orig)

#%%Save
np.save("types_inf_%s_%d" %(depiction,patient),types_inf)  #Types after mc simulation: [types(9000),runs,sample]; need drop 0
np.save("Energy_vec_%s_%d" %(depiction,patient),Energy_vec) #Energy in mc simulation: [runs,n_samples,Energy(10000)]; need drop 0

#%%---Plot mc optimized img of infered parameters
plt.close('all')
patient = 4
run = 1

T = 1.; mcStep = 150
positions = cell_center_data[0:n_cells[0],0:2,0]*0.39; 
uniqueCellTypes = np.unique(cell_center_data[0:n_cells[0],2,0]).astype(int)
n_cellTypes = len(uniqueCellTypes)
uniques = cellTypes[np.unique(cell_center_data[0:n_cells[0],2,0]).astype(int)]
#Local Probability SIMULATION
types_conf = types_inf[0:n_cells[0],run,0]
neighbor_register_conf = mplb.neighbors(positions,types_conf,rlimit[0],boxlength)
P_config = mplb.neighborProb(neighbor_register_conf,types_conf,uniqueCellTypes,cellTypes_orig)
#Local Probability PATHOLOGY
types_patho = cell_center_data[0:n_cells[0],2,0]
neighbor_register_patho = mplb.neighbors(positions,types_patho,rlimit[0],boxlength)
P_patho = mplb.neighborProb(neighbor_register_patho,types_patho,uniqueCellTypes,cellTypes_orig)

#%%

np.save("Pconfig_%s_%d" %(depiction,patient),P_config) 
np.save("Ppatho_%s_%d" %(depiction,patient),P_patho) 
print(f'Sample {patient}; Run {run};'
      +f'\n MPL eval = {evalvec[run,0]}; \n rlimit = {rlimit[0]}; \n avg_neighb = {avg_neighb[0]}'
      +'\n Uniques: ');print(uniques)
print(f'\n Jmat_opt = {(Jmat_optArr[:,:,run,0][Jmat_optArr[:,:,run,0] != 999]).reshape(n_cellTypes,n_cellTypes)}')    

mclb.latticeplt(positions,cell_center_data[0:n_cells[0],2,0],cellTypes,T,mcStep=0)
#PLOT Simulated lattice               
mclb.latticeplt(positions,types_inf[0:n_cells[0],run,0],cellTypes,T,mcStep) 
#PLOT Energy
Energy = Energy_vec[run,0,np.where(Energy_vec[run,0,:]!=0)[0]]
success_steps = np.arange(0,len(Energy))
plt.figure('Energy1')
plt.plot(success_steps,Energy)
plt.suptitle('Energy')
plt.title(f'Patient {patient}; Run {run}')
plt.xlabel('Successfull MC Steps'); plt.ylabel('Energy')

#%% 
mplb.nProbPlot(P_patho,cellTypes,3) #Pathology: Local neighborhood prob
mplb.nProbPlot(P_config,cellTypes,3) #Simulation: Local neighborhood prob

#%% 
