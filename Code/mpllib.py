# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:05:46 2020

@author: alber
"""

import mclib as mclb
import numpy as np
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import axes3d
from numba import jit 
from numba.experimental import jitclass         # import the decorator
from numba import int32, float64    # import the types
from tqdm import tqdm 


#%%
         
def neighbors(positions,immune_tag,rlim,boxlength): 
    """
    Generates an object that stores info about the neighborhood of every cell
    IN: positions: f8[:,2]; array of cell positions in 2D
        immune_tag: i4[:]; list of cell types. Is of same length as positions
        rlim: f8; radius that defines the size of the neighborhood around every cell. 
    OUT: neighbor_register[:].i/.ni/.t/.nt; call a specific cell and access info about its neighborhood
    """
    spec = [
    ('i', float64),               #The cell identity (index)
    ('t', float64),               #The cell type [0,q-1]  
    ('ni', float64[:]),           #List of neighbor identities (indices)
    ('nt', float64[:]),           #List of neighbor types
    ]
    
    @jitclass(spec)
    class neighbor_class:
         def __init__(self,cellid,celltype,ni,nt):
             self.i = cellid   #id of target
             self.t = celltype #type of target
             self.ni = np.array([  ni[k]  for k in range(len(ni)) ])    #neighbor id
             self.nt = np.array([  nt[l]  for l in range(len(nt)) ]) #neighbor type
    
    neighbor_register = [];
    for i in range(0,int(len(positions))):
        n = []; nt = [];
        a = positions[i]
        for j in range(0,len(positions)):
            if (j == i):
                continue
            b = positions[j]
            dx = a[0] - b[0]; dy = a[1] - b[1]
            #dx = dx - boxlength * np.round(dx/boxlength); #minimal image
            #dy = dy - boxlength * np.round(dy/boxlength);
            d = np.sqrt(dx**2 + dy**2)
            if (d <= rlim):
                n  = np.append(n,[j])
                nt = np.append(nt,[immune_tag[j]])           
        neighbor_register = np.append(neighbor_register,neighbor_class(i,immune_tag[i],n,nt))
    return neighbor_register 

@jit('(f8[:])(f8[:],i4)',nopython=True)
def calcProp_jit(types,n_cellTypes):
    """
    Calculate the proportion of cell types in a set of cells
    IN: types: f8[:]; array of cell types 
        n_cellTypes: i4; number of different cell types
    OUT: mType: f8[n_cellTypes]; proportion of different cells.
    """
    L = len(types)
    mType = np.zeros(n_cellTypes)
    for j in range(0,n_cellTypes):
        index = np.where(types == j)
        n_type = len(index[0])/L
        mType[j] = n_type
    return mType

@jit('(f8)(i4,i4,f8[:,:])',nopython=True)
def connectivity(C1, C2,Jmat):
    """
    H_Potts
    """
    return 1.*Jmat[C1,C2]
                       

        
def f_eval(xmg,ymg,parameters,function,argorder):
    """
    xmg, ymg : f8[evalpoints,evalpoints]; meshgrid on which the 2D function is evaluated
    parameters : f8[:]; 1D array of constant parameters. 
    function : function handle; The 2D fct to evaluate. 
        If function takes more than 2 input parameters 2 are varied and the others (parameters) are const

    Returns: z; The evaluation of function on a grid with variation in two parameters (xmg,ymg)
    """
    x_flat = np.ndarray.flatten(xmg)
    y_flat = np.ndarray.flatten(ymg)
    xy = np.array([x_flat,y_flat]).T
    
    
    length = len(xy)
    z = np.zeros(len(xy))
    
    swap1 = argorder #to define the right order for the function input.
    swap2 = [0,0]
    swap2[0] = swap1[1]; swap2[1] = swap1[0]
    
    for i in range(0,length):
        fctarg = np.concatenate((xy[i,:],parameters)) #concatenate constant parameters
        fctarg[swap1] = fctarg[swap2]
        z[i] = function(fctarg)
    z = np.reshape(z,np.shape(xmg))    
    return z

def mpl_eval(xmg,ymg,parameters,argorder,n_cellTypes,neighbor_register,beta):
    """
    xmg, ymg : f8[evalpoints,evalpoints]; meshgrid on which the 2D function is evaluated
    parameters : f8[:]; 1D array of constant parameters. 
    function : function handle; The 2D fct to evaluate. 
        If function takes more than 2 input parameters 2 are varied and the others (parameters) are const

    Returns: z; The evaluation of function on a grid with variation in two parameters (xmg,ymg)
    """
    x_flat = np.ndarray.flatten(xmg)
    y_flat = np.ndarray.flatten(ymg)
    xy = np.array([x_flat,y_flat]).T
    
    
    length = len(xy)
    z = np.zeros(len(xy))
    
    swap1 = argorder #to define the right order for the function input.
    swap2 = [0,0]
    swap2[0] = swap1[1]; swap2[1] = swap1[0]
    
    for i in range(0,length):
        fctarg = np.concatenate((xy[i,:],parameters)) #concatenate constant parameters
        fctarg[swap1] = fctarg[swap2]
        z[i] = mplikelihood(fctarg,n_cellTypes,neighbor_register,beta)
    z = np.reshape(z,np.shape(xmg))    
    return z


def symmat_from_vec(linvec,mat_len): #mat_len is the size of the square matrix symmat 
    """
    Creates a symmetric matrix from a linear vector
    IN: linvec: f8[:]; an np array
        mat_len: f8; the length of the symmetric matrix
    """
    if (len(linvec) == 1):
        symmat = np.zeros((mat_len,mat_len))
        symmat[0,0] = linvec[0]
    elif(len(linvec) == mat_len):
        symmat = np.zeros((mat_len,mat_len))
        np.fill_diagonal(symmat,linvec)
    else:
        symmat = np.zeros((mat_len,mat_len))
        symmat[np.triu_indices(mat_len, 0)] = linvec
        symmat = symmat + symmat.T - np.diag(symmat.diagonal())
    return symmat

       
def mplContoure(xfrom,xto,yfrom,yto,opt_object,n_param,Jmat,neighbor_register,TT):
    """
    Creates, according to the combination n_param free parameters and the length of the
    connectivity matrix Jmat a contour plot of the negative maximum likelihood landscape (-MPL).
    IN: xfrom,-to/ yfrom,-to: f8[3]; Defines the plotting range in x and y direction.
    opt_object: The optimizer object is the result of the general purpose optimisation in 
                the function MPLmaximizer.
    n_param: number of free optimisation parameters (a plot will be generated for n_param < 4)
    Jmat: The connectivity matrix with which the cells are MCMC encoded.
    TT: [T_dec]; array of decoding temperature.
    """
    beta = 1/TT[0]
    n_cellTypes = len(Jmat)
    if (n_param == 1):
        xx = np.linspace(xfrom[0],xto[0],30)
        mpl = np.zeros(len(xx))
        for i in range(0,len(xx)):
            mpl[i] = mplikelihood(np.array([xx[i]]),n_cellTypes,neighbor_register,beta)
        plt.figure()
        plt.plot(xx,mpl)
        plt.axvline(x=xx[np.argmin(mpl)],color='r',ls='--')
        plt.xlabel('J11'); plt.ylabel('-MPL')
        plt.suptitle('-MPL for 2 cell types'); plt.title(f'J12 = J12 = 0')
        
    elif (n_param == 2):
        xx = np.linspace(xfrom[0],xto[0],20) #J11 parameter space
        yy = np.linspace(yfrom[0],yto[0],20) #J12 parameter space
        [xmg,ymg] = np.meshgrid(xx,yy)
        levels = 15 #layer resolution
        argorder_in = np.array([1,2]) #const J12 = 0
        Jmat_opt = np.array([[opt_object.x[0],0.],[0.,opt_object.x[1]]])
        const_param = [0]
        z = mpl_eval(xmg,ymg,const_param,argorder_in,n_cellTypes,neighbor_register,beta)
        ticks = np.linspace(np.floor(np.min(z)), np.ceil(np.max(z)), num=levels, endpoint=True)
        
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(xmg, ymg, z, ticks)
        plt.scatter(Jmat_opt[0,0],Jmat_opt[1,1],s=20,c='r',label='Decoded parameters')
        plt.scatter(Jmat[0,0],Jmat[1,1],s=20,facecolors='none', edgecolors='r',label='Encoded parameters')
        fig.colorbar(cp,ticks=ticks,extendrect='True')
        ax.set_title('-MPLikelyhood; J12 = 0'); ax.set_xlabel(r'J_11'); ax.set_ylabel(r'J_22')
        plt.legend(); plt.show()  
        
    elif (n_param == 3):
        mat_len = len(Jmat)
        Jmat_opt = symmat_from_vec(opt_object.x,mat_len)
        param_enc = Jmat[np.triu_indices(mat_len)]
        triu_indices = np.array([[0,3,1],[1,0,3],[3,0,1]]) #const/dependent meshx /dependent meshy
        
        fig, axs = plt.subplots(1,3, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(wspace=.5)
        axs = axs.ravel()
        fig.suptitle('-MPLikelihood',fontsize=16)
        for i in range(0,n_param):
            argorder = np.array([[0,2],[1,2],[2,2]])
            argorder_in = argorder[i,:].astype(int);
            xx = np.linspace(xfrom[i],xto[i],15) # parameter space
            yy = np.linspace(yfrom[i],yto[i],15) #parameter space
            [xmg,ymg] = np.meshgrid(xx,yy)
            z = mpl_eval(xmg,ymg,np.array([param_enc[i]]),argorder_in,n_cellTypes,neighbor_register,beta)

            c_row,c_column = arrIndexConv(triu_indices[i,0],n_columns=mat_len)
            d1_row,d1_column = arrIndexConv(triu_indices[i,1],n_columns=mat_len)
            d2_row,d2_column = arrIndexConv(triu_indices[i,2],n_columns=mat_len)
            levels = 15
            ticks = np.linspace(np.floor(np.min(z)), np.ceil(np.max(z)), num=levels, endpoint=True); #print(ticks)
            cp = axs[i].contourf(xmg, ymg, z,ticks)
            axs[i].scatter(Jmat_opt[d1_row,d1_column],Jmat_opt[d2_row,d2_column],s=20,c='r',label='Decoded parameters')
            axs[i].scatter(Jmat[d1_row,d1_column],Jmat[d2_row,d2_column],s=20,facecolors='none', edgecolors='r',label='Encoded parameters')
            axs[i].set_title(f'Const J{c_row + 1}{c_column + 1} = {param_enc[i]}'); 
            axs[i].set_xlabel(f'J{int(d1_row)+1}{int(d1_column)+1}'); axs[i].set_ylabel(f'J{int(d2_row)+1}{int(d2_column)+1}')                   
            axs[i].legend()
            fig.colorbar(cp,ax=axs[i],ticks=ticks,extendrect='True'); #cbar.set_ticks(cbarlabels); cbar.set_ticklabels(cbarlabels)
        plt.show()
        
def mplContoure2(xfrom,xto,yfrom,yto,Jmat_opt1,Jmat_opt2,Jmat,evalvec,n_cellTypes,n_param,neighbor_register,TT,S):
    """
    Creates, according to the combination n_param free parameters and the length of the
    connectivity matrix Jmat a contour plot of the negative maximum likelihood landscape (-MPL).
    (See mplContoure). It is possible to display 2 different results of the optimisation
    IN: xfrom,-to/ yfrom,-to: f8[3]; Defines the plotting range in x and y direction.
    Jmat_opt1-2: f8[n_cellTypes,n_cellTypes]; The optimizer infered connectifity matrix 1 & 2
                Note: if Jmat_opt2 = 0 - it is not plotted!
    Jmat: The connectivity matrix with which the cells are MCMC encoded.
    evalvec: f8[3]; The evaluation of mplikelihood() at Jmat_opt1, Jmat_opt2, Jmat parameters
                Note: if evalvec = 0 - no annotations are displayedS
    n_param: number of free optimisation parameters (a plot will be generated for n_param < 4)
    TT: [T_dec]; array of decoding temperature.
    """
    beta = 1/TT[0]
    if (n_param == 1):
        xx = np.linspace(xfrom[0],xto[0],30)
        mpl = np.zeros(len(xx))
        for i in range(0,len(xx)):
            mpl[i] = mplikelihood(np.array([xx[i]]),n_cellTypes,neighbor_register,beta)
        plt.figure()
        plt.plot(xx,mpl)
        plt.axvline(x=Jmat_opt1[0,0],color='r',ls='--')
        plt.xlabel('J11'); plt.ylabel('-MPL')
        plt.suptitle('-MPL for 2 cell types'); plt.title(f'J12 = J12 = 0')
        
    elif (n_param == 2):
        xx = np.linspace(xfrom[0],xto[0],22) #J11 parameter space
        yy = np.linspace(yfrom[0],yto[0],22) #J12 parameter space
        [xmg,ymg] = np.meshgrid(xx,yy)
        levels = 15 #layer resolution
        argorder_in = np.array([1,2]) #const J12 = 0
        const_param = [0]
        z = mpl_eval(xmg,ymg,const_param,argorder_in,n_cellTypes,neighbor_register,beta)
        ticks = np.linspace(np.floor(np.min(z)), np.ceil(np.max(z)), num=levels, endpoint=True)
        
        fig,ax=plt.subplots(1,1)
        cp = ax.contourf(xmg, ymg, z, ticks)
        if (type(Jmat_opt1) != int):
            plt.scatter(Jmat_opt1[0,0],Jmat_opt1[1,1],s=20,c='r',label='1DOF-Decoded parameters')
        if (type(Jmat) != int):
            plt.scatter(Jmat[0,0],Jmat[1,1],s=20,facecolors='none', edgecolors='maroon',label='Encoded parameters')
        if (type(Jmat_opt2) != int):
            plt.scatter(Jmat_opt2[0,0],Jmat_opt2[1,1],s=20,c='brown',label='2DOF-Decoded parameters')
            x = [Jmat_opt1[0,0],Jmat_opt2[0,0],Jmat[0,0]]; y = [Jmat_opt1[1,1],Jmat_opt2[1,1],Jmat[1,1]]
            if type(evalvec) != int : 
                for i, txt in enumerate(evalvec):
                    ax.annotate(txt, (x[i], y[i]))
        fig.colorbar(cp,ticks=ticks,extendrect='True')
        plt.suptitle('-MPLikelyhood'); ax.set_title(f'J12 = 0; S = {S:.3}');
        ax.set_xlabel(r'J_11'); ax.set_ylabel(r'J_22')
        if (type(Jmat) != int): 
            plt.legend(loc='best');
        plt.show()  
        
    elif (n_param == 3):
        mat_len = len(Jmat)
        param_enc = Jmat[np.triu_indices(mat_len)]
        triu_indices = np.array([[0,3,1],[1,0,3],[3,0,1]]) #const/dependent meshx /dependent meshy
        
        fig, axs = plt.subplots(1,3, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(wspace=.5)
        axs = axs.ravel()
        fig.suptitle('-MPLikelihood',fontsize=16)
        for i in range(0,n_param):
            argorder = np.array([[0,2],[1,2],[2,2]])
            argorder_in = argorder[i,:].astype(int);
            xx = np.linspace(xfrom[i],xto[i],15) # parameter space
            yy = np.linspace(yfrom[i],yto[i],15) #parameter space
            [xmg,ymg] = np.meshgrid(xx,yy)
            z = mpl_eval(xmg,ymg,np.array([param_enc[i]]),argorder_in,n_cellTypes,neighbor_register,beta)

            c_row,c_column = arrIndexConv(triu_indices[i,0],n_columns=mat_len)
            d1_row,d1_column = arrIndexConv(triu_indices[i,1],n_columns=mat_len)
            d2_row,d2_column = arrIndexConv(triu_indices[i,2],n_columns=mat_len)
            levels = 15
            ticks = np.linspace(np.floor(np.min(z)), np.ceil(np.max(z)), num=levels, endpoint=True); #print(ticks)
            cp = axs[i].contourf(xmg, ymg, z,ticks)
            axs[i].scatter(Jmat_opt1[d1_row,d1_column],Jmat_opt1[d2_row,d2_column],s=20,c='r',label='1DOF-Decoded parameters')
            if (type(Jmat_opt2) != int):
                axs[i].scatter(Jmat_opt2[d1_row,d1_column],Jmat_opt2[d2_row,d2_column],s=20,c='brown',label='2DOF-Decoded parameters')
            axs[i].scatter(Jmat[d1_row,d1_column],Jmat[d2_row,d2_column],s=20,facecolors='none', edgecolors='maroon',label='Encoded parameters')
            axs[i].set_title(f'Const J{c_row + 1}{c_column + 1} = {param_enc[i]}; S = {S:.3}'); 
            axs[i].set_xlabel(f'J{int(d1_row)+1}{int(d1_column)+1}'); axs[i].set_ylabel(f'J{int(d2_row)+1}{int(d2_column)+1}')                   
            axs[i].legend(loc='upper left')
            fig.colorbar(cp,ax=axs[i],ticks=ticks,extendrect='True'); #cbar.set_ticks(cbarlabels); cbar.set_ticklabels(cbarlabels)
        plt.show()
       
            
    
def arrIndexConv(list_index,n_columns):
    """
    Converts list indices of a 2D array to row and col indices
    IN: list_index i4; list index of 2D array
        n_columns: i4; number of columns of the array
    OUT: row, colum: i4,i4; row and column index of the 2D array 
    """
    colum = int(list_index % n_columns)
    row = int(list_index / n_columns)
    return row, colum
        
def locEnergy(targetI,neighbor_register,Jmat,beta):
   """
   targetI : i4; Target identity.
   neighbor_register : neighbor_class[n_cells]; i-target identity/ t-target type/ 
                       ni-neighbor identity (vector)/ nt-neighbor type (vector)
   Jmat: f8[n_cellTypes,n_cellTypes]; symmetric connectivity matrix
   beta : f8; MC inv Temperature
   Returns: exp(beta H_s): exponential of local energy
   """
   n_neighbors = len(neighbor_register[targetI].ni)
   targetT = neighbor_register[targetI].t
   H = 0
   for l in range(0,n_neighbors): #loop over neighbors
       neighborT = neighbor_register[targetI].nt[l] #cell type of targets neighbor
       H += connectivity(int(targetT),int(neighborT),Jmat) 
   return -beta * H
        
        

def locPartition(targetI,neighbor_register,n_cellTypes,Jmat,beta):
    """
    targetI : i4; Target identity.
    neighbor_register: neighbor class; info about a cell and its neighbors
    cellTypes : i[:]/ or str[:]; Vector of different cell types
    Jmat: f8[n_cellTypes,n_cellTypes]; symmetric connectivity matrix
    beta : f8; MC inv Temperature
    Returns: Ln of local (Neares Neighbor) partition evaluation
    """    
    n_neighbors = len(neighbor_register[targetI].ni)
    P = np.zeros(n_cellTypes)
    for k in range(0,n_cellTypes): #loop over cell types
        targetT = k
        H = 0
        for l in range(0,n_neighbors): #loop over neighbors
            neighborT = neighbor_register[targetI].nt[l] #cell type of targets neighbor
            H += connectivity(int(targetT),int(neighborT),Jmat);   
        P[k] = -beta*H
    return sp.special.logsumexp(P)  
    
 
def mplikelihood(Jparam,n_cellTypes,neighbor_register,beta):#Sanming Song et al. IEEE 2016 'Local Autoencoding...' eq.(6)
    """
    Evaluates the negative maximum pseudo likelihood for a given graph (positions->neighbor_register)
    It analyses 2 cell types. While J12 & J22 = 0 it inferes the most likely value for J11.
    This is enough to create sorting/mix/randome scenario.
    IN: Jparam: f8; parameter for J11
    OUT: -mpl_glob: f8; negative maximal pseudo likelihoodd for a given graph and connectivity matrix
    """
    if (len(Jparam) == 1): #infere J11
        Jmat = np.zeros((n_cellTypes,n_cellTypes))
        Jmat[0,0] = Jparam[0]
    elif (n_cellTypes == len(Jparam)): #infere diag elements only
          Jmat = np.zeros((n_cellTypes,n_cellTypes))
          np.fill_diagonal(Jmat,Jparam)      
    else:
        Jmat = symmat_from_vec(Jparam,n_cellTypes) #infere all elements
    mpl_glob = 0 #Global maximum pseudo likelihood
    for k in range(0,len(neighbor_register)): #loop over all sights/ scan through image
        targetI = k
        H = locEnergy(targetI,neighbor_register,Jmat,beta)
        Z = locPartition(targetI,neighbor_register,n_cellTypes,Jmat,beta)
        mpl_glob += (H-Z); #with gaussian prior: mu=0, var=1
    return -(mpl_glob-0.5*np.sum(Jparam**2))

def mplMaximizer(random_walks,function,n_param,n_cellTypes,neighbor_register,TT): 
    """
    random_walks : f8; number of random walks, i.e different random init guesses
    function : function handle; function to optimize: sp.optimize.minimize(...,method='Nelder-Mead')
    n_param : f8; number of free parameters to optimize

    Returns: opt_object: .x->optimal parameters; .fun-> fct evaluation at opt parameters
    """
    #n_cellTypes is used in mplikelihood
    for k in range(0,len(TT)):
        beta = 1/TT[k]
        
        minimum = 9e15
        for i in tqdm(range(0,random_walks),position=0,desc ='MPL Maximizer'): #Minimize for different initial guesses
            initial_guess = np.random.uniform(-5,5,n_param); #print(initial_guess) #init guess for values of connectivity matrix Jmat
            optimizer_object = optimize.minimize(function, initial_guess, #optimize for Jparam->connectivity matrix
                                                    args=(n_cellTypes,neighbor_register,beta),method='Nelder-Mead',
                                                    options={'maxiter':len(initial_guess)*400,'disp':True,'return_all':True}) #args are const parameters
            y = optimizer_object.fun
            if (y < minimum): #keep parameters if -MPL is smaller than before
                minimum = y;# print(minimum)
                opt_object = optimizer_object; 
        if (n_param == 1): #infere J11 & J22
            Jmat_opt = np.array([[opt_object.x[0],0.],[0.,0]])
        elif (n_cellTypes == len(opt_object.x)): #infere diag elements only
              Jmat_opt = np.zeros((n_cellTypes,n_cellTypes))
              np.fill_diagonal(Jmat_opt,opt_object.x)   
        else:
            Jmat_opt = symmat_from_vec(opt_object.x,n_cellTypes)     
        print(f'\nParameters that minimize the MPL: \nJmat_opt = {Jmat_opt}; MPL-T = {TT[k]} \n')
    return opt_object     
    
    
def jmatTrans(Jmat_opt,threshold):
    n_cellTypes = len(Jmat_opt)
    Jmat_trans = np.zeros((n_cellTypes,n_cellTypes))
    for i in range(0,len(Jmat_opt)):
        for j in range(0,len(Jmat_opt)):
            value = Jmat_opt[i,j]-Jmat_opt[i,i]
            if np.abs(value) <= threshold:
                value = 0
            Jmat_trans[i,j] = value
    return Jmat_trans    
    
    
def patternCalc(n_cellTypes):
    """
    Calculates how many different spacial patterns (äquivalence classes) are possible
    to form.
    IN: n_cellTypes: i4; Number of different cell Types
    """
    n_pattern = 0
    if n_cellTypes > 2:
        for i in range(1,n_cellTypes-1):
            n_pattern += sp.special.binom(n_cellTypes,i)
    n_pattern = 2*(n_pattern)+3
    return int(n_pattern)    

def neighborProb(neighbor_register,types_end,uniqueCellTypes,cellTypes_orig):
    """
    Calculates, for a given image (graph), the probability of finding a cell type 
    in the neighborhood of the reference type.
    IN: neighbor_register: object
        types_end: i4[n_cells]: list of cell types (after MCMC optimisation)
        n_cellTypes: i4; Number of different cell types
    OUT: glob_p: f8[n_cellTypes,n_cellTypes]; Matrix that contains the probabilities
        of finding a certain cell type in the neighborhood of its another type.
        Note: The diagonal elements are the probability of finding its own type 
        in a certain neighborhood.
    """
    n_cellTypes = int(len(uniqueCellTypes))
    #proportions = calcProp_jit(types_end,n_cellTypes)
    glob_p = np.zeros((cellTypes_orig[0],cellTypes_orig[0]))
    for i in range(0,n_cellTypes):
        type_index = np.where(types_end == uniqueCellTypes[i])[0]
        for j in range(0,len(type_index)):
            loc_p = np.zeros(cellTypes_orig[0])
            for k in range(0,n_cellTypes):
                loc_p[uniqueCellTypes[k]] = len(np.where(neighbor_register[type_index[j]].nt==uniqueCellTypes[k])[0])/len(neighbor_register[type_index[j]].nt)
            glob_p[uniqueCellTypes[i],:] += loc_p
        glob_p[uniqueCellTypes[i],:] = glob_p[uniqueCellTypes[i],:]/len(type_index)
    return glob_p
   
    
def nProbPlot(glob_p,cellTypes,i):
    """
    Plots the probability of finding a cell type within a neighborhood of 
    cell type i.
    IN: glob_p: f8[n_cellTypes,n_cellTypes]; Matrix that contains the probabilities
    cellTyles: srt704[n_cellTypes]; Names of cell types
    n_cellTypes: i4; number of different cell types.
    i: i4;  reference cell type in question
    """
    n_cellTypes = int(len(glob_p))
    indices = np.arange(n_cellTypes)
    glob_pi = glob_p[i,:]
    colors = plt.cm.jet(np.linspace(0,1, n_cellTypes))
    plt.figure()
    plt.bar(indices,glob_pi,width=0.3,color=colors)
    plt.xticks(indices,cellTypes,fontsize=12, rotation=90)
    plt.ylabel('P(Cell Type)')
    plt.title('Probability of cells within a neighborhood of '+cellTypes[i])
    plt.show()
    
def nProbPlot1(glob_p,cellTypes,n_cellTypes,i):
    n_cellTypes = int(n_cellTypes)
    indices = np.arange(n_cellTypes)
    glob_pi = rightRotate(glob_p[i,:],n_cellTypes-i)
    plt.figure()
    plt.bar(1,glob_pi[0],color='k',label=cellTypes[i])
    bottom=0
    for j in range(0,n_cellTypes-1):
        plt.bar(j,glob_pi[j+1],bottom=bottom,label=cellTypes[rightRotate(indices,n_cellTypes-i-1)[j]])
        bottom += glob_pi[j+1] 
    
    plt.ylabel('P(Cell Type)')
    plt.title('Probability of cells within a neighborhood of '+cellTypes[i])
    plt.legend(); plt.show()
        
    
def rightRotate(lists, num): 
    """
    Executes a right rotation of the values from lists num of times.
    lists: list[:] of integers, floats or strings
    num: i4; number of rotations to the right
    """
    output_list = [] 
 
    # Will add values from n to the new list 
    for item in range(len(lists) - num, len(lists)): 
        output_list.append(lists[item]) 
      
    # Will add the values before 
    # n to the end of new list     
    for item in range(0, len(lists) - num):  
        output_list.append(lists[item]) 
          
    return output_list     
    
def propagationPlt(S,beta,neighbors):
    """
    Generates a propagation plot
    IN: S: f8[n_evalpoints,n_sampeles]; property in question
        beta: f8[n_evalpoints,n_samples]; evaluation domain of property 
        neighbors: f8[n_samples]; number of discrete sample measurements
    """
    verts = []
    for irad in range(len(neighbors)):
        # I'm adding a zero amplitude at the beginning and the end to get a nice
        # flat bottom on the polygons
        xs = np.concatenate([[beta[0,irad]], beta[:,irad], [beta[-1,irad]]])
        ys = np.concatenate([[0],S[:,irad],[0]])
        verts.append(list(zip(xs, ys)))
    
    poly = PolyCollection(verts, facecolors=['r','g','c','m'], edgecolors=(0,0,0,0))
    poly.set_alpha(0.7)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # The zdir keyword makes it plot the "z" vertex dimension (radius)
    # along the y axis. The zs keyword sets each polygon at the
    # correct radius value.
    ax.add_collection3d(poly, zs=neighbors, zdir='y')
    
    ax.set_xlim3d(beta.min(), beta.max())
    ax.set_xlabel(r'$\beta \cdot J$')
    ax.set_ylim3d(neighbors.min(), neighbors.max())
    ax.set_ylabel('#Neighbors')
    ax.set_zlim3d(0, S.max())
    ax.set_zlabel('S')
    plt.title('Entropy')
    plt.show()    
  
    
    
def paramInference(cell_center_data,runs,limiter,TT_dec,boxlength):
   samples = [0]
   random_walks = 2 
   #number of independent entries in the symmetric matrix Jmat// depending on n_cellTypes
   n_samples = len(samples)
   rlimit = np.zeros(n_samples)
   n_cells = np.zeros(n_samples,dtype=int)
   avg_neighb = np.zeros(n_samples)
   evalvec = np.zeros((runs,n_samples))
   Jmat_optArr = 999*np.ones((110,110,runs,n_samples)) #preallocate wit 999 (needs clearing in mcGenerator())
   for k in range(0,n_samples): #patient samples
       patient = samples[k] #patient under consideration
       positions = cell_center_data[:,0:2,patient]*0.39; #Positions & cell types:
       positions = positions[np.where(positions[:,0]!=0)[0],:]; #clear 0 preallocation
       if limiter != False: positions = positions[0:limiter,:] #if true: limit to cells under consideration
       n_cells[k] = int(len(positions))
       n_cellTypes = len(np.unique(cell_center_data[0:n_cells[k],2,patient]))
       n_param = int((n_cellTypes*(n_cellTypes+1))/2)
       types_end = cell_center_data[:,2,patient]; types_end = types_end[0:n_cells[k]]
       
       rlimit[k] = mclb.minDistance(positions) #Neighborhood
       neighbor_register = neighbors(positions,types_end,rlimit[k],boxlength)
       avg_neighb[k] = mclb.avgneighb(neighbor_register)
       for i in range(0,runs): #infering parameters
           print(f'\nDecoding: Sample {patient}; Run {i+1}/{runs}\n')
           try:
               opt_object = mplMaximizer(random_walks,mplikelihood,n_param,n_cellTypes,neighbor_register,TT_dec)
               evalvec[i,k] = opt_object.fun
               Jmat_opt = symmat_from_vec(opt_object.x, n_cellTypes)
           except:
               evalvec[i,k] = 0
               Jmat_opt = np.zeros((n_cellTypes,n_cellTypes))
               print(Jmat_opt)

           Jmat_optArr[0:len(Jmat_opt),0:len(Jmat_opt),i,k] = Jmat_opt #measuring the data
   return Jmat_optArr,evalvec,rlimit,avg_neighb,n_cells

    


