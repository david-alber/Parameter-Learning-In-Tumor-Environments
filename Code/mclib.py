# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:44:20 2020

@author: alber
"""
import mpllib as mplb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import jit 
from numba.experimental import jitclass          # import the decorator
from numba import int32, float64, void    # import the types 
from tqdm import tqdm  
#%%

def discrete_cmap(N):
    """
    Define a discrete color map for scatter plot
    IN: N - number of different colors
    """
    # Discrete coller map plotting routine:
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, bounds

def latticeplt(positions,types,cellTypes,T,mcStep):
    n_cell_types = len(cellTypes)
    cmap, norm, bounds = discrete_cmap(n_cell_types)
    # setup the plot
    #plt.figure()
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    # make the scatter
    img = ax.scatter(positions[:,0],positions[:,1],c=types,s=4,cmap=cmap,norm=norm)
    # create the colorbar
    cb = plt.colorbar(img, spacing='proportional',ticks=bounds)
    cb.set_ticklabels(cellTypes)
    cb.set_label('Cell Types')
    fig.suptitle('Potts model simulation of tumor tissue')
    #ax.set_title('Scale: 1:0.39 micrometer')
    ax.set_xlabel(r'x [$\mu m$]'); ax.set_ylabel(r'y [$\mu m$]')
    plt.suptitle('Lattice of computation domain')
    plt.title(f'T = {T:.2}; MCS = {mcStep}')
    #plt.gca().invert_xaxis() #invert x axis to match plotImg
    plt.show()
    
@jit('(f8)(i4,i4,f8[:,:])',nopython=True)
def connectivity(C1, C2,Jmat):
    """
    H_Potts 
    """
    return 1.*Jmat[C1,C2]

spec = [
    ('t', int32),
    ('x', float64),
    ('y', float64),
]

@jitclass(spec)
class cell_class1:
     def __init__(self,celltype,x,y):
         self.t = celltype
         self.x = x
         self.y = y

#@jit('Tuple((f8,f8[:]))(,f8,f8)',nopython=True)
def ERList_class(cell_register,Jmat,rlimit,boxlength):
    n_particles = len(cell_register)
    Energy = 0
    rlist = np.zeros(int((n_particles-1)*n_particles/2)) #Preallocate rlist// np.zeros() is not in jit library
    count = 0
    for i in range(0,n_particles-1):      #particle loop
        for j in range(i+1,n_particles):  #compare to loop
            dx = (cell_register[i].x - cell_register[j].x);
            dy = (cell_register[i].y - cell_register[j].y);

            #dx = dx - boxlength * np.round(dx/boxlength); #minimal image
            #dy = dy - boxlength * np.round(dy/boxlength);
            
            r = np.sqrt(dx**2 + dy**2)
            if (r <= rlimit):
                C1,C2 = cell_register[i].t, cell_register[j].t
                Energy += connectivity(C1,C2,Jmat)
                rlist[count] = r #store radii for rdf
                count += 1
    Energy = Energy 
    return Energy, rlist 

@jit('Tuple((f8[:],f8[:,:]))(f8[:,:],f8[:],f8[:,:],f8,f8)',nopython=True)
def ERList_arr(positions,types,Jmat,rlimit,boxlength):
    n_particles = len(positions)
    Energy = 0.
    H_mat = np.zeros((n_particles,n_particles)) #upper triang matrix with energy contributions
    count = 0
    for i in range(0,n_particles-1):      #particle loop
        for j in range(i+1,n_particles):  #compare to loop
            dx = (positions[i,0] - positions[j,0]);
            dy = (positions[i,1] - positions[j,1]);

            #dx = dx - boxlength * np.round(dx/boxlength); #minimal image
            #dy = dy - boxlength * np.round(dy/boxlength);
            
            r = np.sqrt(dx**2 + dy**2)
            if (r <= rlimit):
                C1,C2 = types[i], types[j]
                e = connectivity(C1,C2,Jmat)
                Energy += e
                H_mat[i,j] = e #store radii for rdf
                count += 1        
    Energy = Energy
    return np.array([Energy]), H_mat

@jit('Tuple((f8[:],i8[:,:]))(f8[:,:])',nopython=True)
def distCalc(positions):
    n_particles = len(positions)
    D = np.zeros((n_particles,n_particles))
    for i in range(0,n_particles-1):      #particle loop
        for j in range(i+1,n_particles):  #compare to loop
            dx = (positions[i,0] - positions[j,0]);
            dy = (positions[i,1] - positions[j,1]);

            #dx = dx - boxlength * np.round(dx/boxlength); #minimal image
            #dy = dy - boxlength * np.round(dy/boxlength);
            r = np.sqrt(dx**2 + dy**2)
            D[i,j] = r
            
    D = D + D.T - np.diag(np.diag(D))#symmetrize 
    m = D.shape[0]
    idx = ((np.arange(1,m+1) + (m+1)*np.arange(m-1).reshape(m-1,-1)).reshape(m,-1))
    D = D.reshape(1,-1)[0]
    return D,idx

def minDistance(positions):
    """
    Calculate the limiting radius for randomly distributed positions
    s.t every cell has as little neighbors as possible, but at least one neighbor.
    (See distanceCalc -> jit optimized)
    IN: positions: [n_samples,2]; positions in 2D
    OUT: rlimit: f8; limiting radius. 
    """
    D,idx = distCalc(positions)
    D = D[idx] #this operation is not possible with numba jit
    min_to_i = np.amin(D,axis=1)
    rlimit = np.amax(min_to_i)
    return rlimit

def avgneighb(neighbor_register):
    n_cells = len(neighbor_register)
    neighb_avg = 0
    for i in range(0,n_cells):
        neighb_avg += len(neighbor_register[i].nt)  
    return neighb_avg/n_cells #average number of neighbors


@jit('i8[:](f8[:,:],i8,f8)',nopython=True)
def localNeighborhood(positions,target,rlimit):
    locN = np.array([999]) #preallocate with some number// otherwise type not understood
    for j in range(0,len(positions)):
        if j == target: continue
        dx = (positions[target,0] - positions[j,0]);
        dy = (positions[target,1] - positions[j,1]);
        #dx = dx - boxlength * np.round(dx/boxlength); #minimal image
        #dy = dy - boxlength * np.round(dy/boxlength);
        r = np.sqrt(dx**2 + dy**2)
        if r <= rlimit:
            locN= np.append(locN,j)
    locN = locN[1:]
    return locN




@jit('i8(f8[:,:],i8,i8[:])',nopython=True)
def targetfinder(positions,target,locN):
    n_cells = len(positions)
    localNeighborhood = np.append(target,locN); 
    choose = np.delete(np.arange(0,n_cells),localNeighborhood); 
    s = np.random.choice(choose)
    return s 


@jit('f8(i4,f8[:],f8[:,:],i8[:])',nopython=True)
def EtoNeighbors(target,types,Jmat,locN):
    Energy = 0.
    for j in range(0,len(locN)):
        C1,C2 = types[target], types[locN[j]];
        Energy += connectivity(C1,C2,Jmat)
    return Energy

@jit('Tuple((f8,f8[:]))(f8[:,:],f8[:],f8,f8[:,:],f8,f8)',nopython=True)
def Ediff(positions,types,n_cellTypes,Jmat,rlimit,boxlength):
    s = np.random.randint(0,len(positions)) #choose cells to flip types
    locNs = localNeighborhood(positions,s,rlimit); 
    t = targetfinder(positions,s,locNs); 
    locNt = localNeighborhood(positions,t,rlimit);  
    #Compute neighborhood energy of the two cells
    dEs_1 = EtoNeighbors(s,types,Jmat,locNs);     
    dEt_1 = EtoNeighbors(t,types,Jmat,locNt);     
    dE1 = (dEs_1+dEt_1) #average the energy of the two cells
    #Trail move: Flip the types of the two cells
    typestrail = np.zeros(len(types)) #preallocate to avoid pointing error (change types with typestrail)
    typestrail[:] = types; typestrail[s] = types[t]; typestrail[t] = types[s]  
    #Compute neighborhood energy of fliped cells
    dEs_2 = EtoNeighbors(s,typestrail,Jmat,locNs);    
    dEt_2 = EtoNeighbors(t,typestrail,Jmat,locNt);    
    dE2 = (dEs_2+dEt_2)
    
    dE = dE2-dE1
    return dE,typestrail

@jit('Tuple((f8[:],f8[:,:],f8[:]))(f8[:,:],f8[:],f8,f8[:],f8[:,:],f8,f8,f8)',nopython=True)
def mcUpdate(positions,types,n_cellTypes,Energy,Jmat,rlimit,boxlength,beta):
    n_particles = len(positions)
    types_current = types
    for k in range(0,n_particles):
        dE,typestrail = Ediff(positions,types_current,n_cellTypes,Jmat,rlimit,boxlength)

        if(np.random.random() <= np.exp(-beta*dE)):

            types_current = typestrail
            Energy = np.append(Energy,Energy[-1]+dE) 
    return types_current,positions,Energy

#@jit('Tuple((f8[:,:],f8[:],f8,i4))(f8[:],i4,f8[:,:],f8[:],f8,f8[:,:],f8,f8)',nopython=True)
def contPotts(TT,N,positions,ctypes0,n_cellTypes,Jmat,rlimit,boxlength): #positions,types,n_cellTypes,Jmat,rlimit,boxlength,beta
    Tsteps = len(TT)
    for k in range(0,Tsteps) : #loop over different temperatures
        mcStep = 0      
        T = TT[k]; beta = 1./T
        Energy_0,_ = ERList_arr(positions,ctypes0,Jmat,rlimit,boxlength); #absolute system energy
        for i in tqdm(range(N),position=0,desc =f'MCMC at T = {T:.2}'):
            ctypes1, positions,Energy = mcUpdate(positions,ctypes0,n_cellTypes,Energy_0,Jmat,rlimit,boxlength,beta)
            mcStep += 1
            ctypes0 = ctypes1
            Energy_0 = Energy
            
    return positions,ctypes1,Energy,T,mcStep

def JmatTrafo(Jmat_optArr,uniques,run,p,cellTypes_orig,n_cellTypes):
    J = (Jmat_optArr[:,:,run,p][Jmat_optArr[:,:,run,p] != 999]).reshape(n_cellTypes,n_cellTypes)

    JJ = np.zeros((cellTypes_orig[p],cellTypes_orig[p]))

    flat_J = np.ndarray.flatten(J)
    xmg,ymg = np.meshgrid(uniques,uniques)
    x_flat = np.ndarray.flatten(xmg)
    y_flat = np.ndarray.flatten(ymg)
    xy = np.array([x_flat,y_flat]).T
    xy = xy.astype(int)
    for i in range(0,len(xy)):
        JJ[xy[i][0],xy[i][1]] = flat_J[i]
    return JJ 

def mcGenerator(Jmat_optArr,cell_center_data,rlimit,runs,N,limiter,TT_enc,boxlength,cellTypes_orig):
    samples = [0]
    n_samples = len(samples)
    Energy_vec = np.zeros((runs,n_samples,9000000))
    print('MC Evaluation: Encode Connectivity \n')
    types_inf = 999*np.ones((9000,runs,n_samples)) #types,runs,sample
    for k in range(0,n_samples):
        patient = samples[k] #patient under consideration
        positions = cell_center_data[:,0:2,patient]*0.39; #Positions & cell types:
        positions = positions[np.where(positions[:,0]!=0)[0],:]; #clear 0 preallocation
        if limiter != False: positions = positions[0:limiter,:] #if true: limit to cells under consideration
        n_cellTypes_orig = cellTypes_orig[0]
        uniques = np.unique(cell_center_data[0:len(positions),2,patient])
        n_cellTypes = len(uniques)
        types_from_img = cell_center_data[0:len(positions),2,patient];
        for i in range(0,runs):
            #Use cell positions & proportions from original img
            propType = mplb.calcProp_jit(types_from_img,n_cellTypes_orig)
            types_init = np.random.choice(np.arange(0,n_cellTypes_orig), size=(len(positions),), p=propType);
            types_init = types_init.astype(float) 
            Jmat = JmatTrafo(Jmat_optArr,uniques,i,k,cellTypes_orig,n_cellTypes) #use infered parameters
            #latticeplt(positions,types_init,cellTypes,TT_enc[0],mcStep=0) #plot the lattice of the initial random field 
            positions,types_end,Energy,T,mcStep = contPotts(TT_enc,N,positions,types_init,
                                                                 n_cellTypes,Jmat,rlimit[k],boxlength)
            types_inf[0:len(positions),i,k] = types_end
            Energy_vec[i,k,0:len(Energy)] = Energy
            
    return types_inf,Energy_vec,T,mcStep
