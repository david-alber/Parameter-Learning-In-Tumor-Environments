# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:56:44 2020

@author: alber
"""
#import pandas as pd 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import jit          
from matplotlib import cm
from skimage import io

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

def pathology1(cell_data,patient,depiction,plotArr,plotImg):
    """
    Creates a pathological image & numpy array of the tumor tissue pathology. 
    The .tiff file of the corresponding patient has to be included in the working directory
    IN: cell_data - Data of cell classification (Keren et al cell:2018)
        patient - int(1-41)
        depiction- "CellGroup" Depiction cell groups/ "ImmuneGroup" Depiction of immune groups
        plot - True/Fales: outputs a plot
    OUT: cell_centers -[n_cells,2] x-y position of cell (scale: 1pixel= 0.39micrometer)  
         immuneGroup - [ncells]
    """
    img = io.imread('p'+str(int(patient))+'_labeledcellData.tiff') #read grayscale values from .tiff image files 
    #out of all the patientsfind the indices for patient in question
    p_files = np.where(cell_data[:,0]==patient)[0] 
    if (depiction == 'CellGroup'):
        #cell goups for patient in question
        immuneGroup = cell_data[p_files,54] - 1  #excel labeling starts with 1-python 0
        cellTypes = np.array(['Undefined','Immune','Endothelial','Mesenchymal-like','Tumor','Kreatin-positive tumor'])   
        mycmap = np.array([[0.5,0.5,0.5],[0.5,0,0.5],[1,0,1],[0.8,0,0.8],[1,0,0],[1,0.5,0]])
        n_cell_types = len(mycmap) #Number of different cell types
    
    elif (depiction == 'ImmuneGroup'):
        immuneGroup = cell_data[p_files,56]   #immunge goups for patient in question
        cellTypes = np.array(['Tumor','Treg','CD4-T','CD8-T','CD3-T','NK','B','Neutrophils','Macrophages','DC','DC/Mono','Mono/Neu','Other Immune'])   
        mycmap = np.array([[0.5,0.5,0.5],[0.5,0,0.5],[1,0,1],[0.8,0,0.8],[1,0,0],[1,0.5,0],[1,1,0],
                           [0.5,1,0],[0,1,0],[0.6,1,0.6],[0,1,1],[0.4,0.4,1],[0,0,0.8]])
        n_cell_types = len(mycmap) #Number of different cell types
        
    cellLabelInImg = cell_data[p_files,1] #cellIDs for patient in question = grayscale expression
    n_cells= len(cellLabelInImg) #Number of distinct cells for every patient
    
    #create create cmap for plotImg
    cmap = plt.cm.gray
    norm = plt.Normalize(img.min(), img.max())
    rgba = cmap(norm(img))

    cell_centers = np.zeros((n_cells,2)) #preallocate for cells center array
    
    #color the cells according to immungeGroup (PlotImg)
    for i in range(0,n_cells): 
        cellLoc = np.where(img == cellLabelInImg[i]) #locate the cell according to grayscale/cell- ID
        x = cellLoc[0][:] #rows of cell
        y = cellLoc[1][:] #col of cell
        rgba[x, y, :3] = mycmap[int(immuneGroup[i])] #color the cell
        
        x_center = np.mean(x)
        y_center = np.mean(y)
        cell_centers[i,:] = [x_center,y_center]
 # PlotArr   
    if(plotArr == True):        
        tag = immuneGroup[0:n_cells]
        # create discrete cmap
        cmap, norm, bounds = discrete_cmap(n_cell_types)
        # setup the plot
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        # make the scatter
        scat = ax.scatter(cell_centers[:,0],cell_centers[:,1],c=tag,s=2.5,cmap=cmap,norm=norm)
        # create the colorbar
        cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
        cb.set_ticklabels(cellTypes)
        cb.set_label('Cell Types')
        fig.suptitle('Pathological image of tumor tissue. P_'+str(int(patient)))
        ax.set_title('Scale: 1:0.39 micrometer')
        plt.gca().invert_yaxis() #invert y axis to match plotImg
        #plt.gca().invert_xaxis() #invert x axis to match plotImg
        plt.show()
# PlotImg 
    if (plotImg == True) :   
        plt.figure()
        plt.imshow(rgba, interpolation='nearest')
        plt.suptitle('Pathological image of tumor tissue. P_'+str(int(patient)))
        plt.title('Scale: 1:0.39 micrometer')
        plt.show()
        
    return   cell_centers, immuneGroup, cellTypes 


#==============================================================================
@jit('Tuple((f8,f8[:]))(f8[:,:],f8)',nopython=True)
def ERList(positions,boxlength):
    n_particles = np.shape(positions)[0]
    eps = 1 #Parameters for energy
    sig = 1
    Energy = 0
    rlist = np.zeros(int((n_particles-1)*n_particles/2)) #Preallocate rlist// np.zeros() is not in jit library
    count = 0
    for i in range(0,n_particles-1):      #particle loop
        for j in range(i+1,n_particles):  #compare to loop
            x = (positions[i,0] - positions[j,0]);
            y = (positions[i,1] - positions[j,1]);

            x = x - boxlength * np.round(x/boxlength); #minimal image
            y = y - boxlength * np.round(y/boxlength);
            
            r = np.sqrt(x**2 + y**2)
            Energy += 4*eps*((sig/r)**12-((sig/r)**6)) 
            rlist[count] = r #store radii for rdf
            count += 1
    Energy = Energy/n_particles
      
    return Energy, rlist 

def blocking(tt,vector,blocks):
        """
        This is a function which helps to process big data files more easily
        by the method of block averaging. 
        For this the first argument is a vector with data, e.g. averaged temperature
        the second argument is another vector, e.g. time grid. 
        The third argument should be the number of blocks. 
        The more blocks, the more data points are taken into consideration. 
        If less blocks, more averaging takes place.
        """
        blockvec = np.zeros(blocks)
        elements = len(vector) 
        rest     = elements % blocks
        if rest != 0: #truncate vector if number of blocks dont fit in vector
            vector   = vector[0:-rest]
            tt       = tt[0:-rest]
            elements = len(vector)   
        meanA  = np.mean(vector)        
        bdata  = int((elements/blocks))#how many points per block
        sigBsq = 0; 
        for k in range(0,blocks):
            blockvec[k] = np.average(vector[k*bdata : (k+1)*bdata]) 
            sigBsq      = sigBsq + (blockvec[k]-meanA)**2    
        sigBsq *= 1/(blocks-1); 
        sigmaB = np.sqrt(sigBsq)
        error  = 1/np.sqrt(blocks)*sigmaB
        blocktt = tt[0:-1:bdata]
        return(blockvec,blocktt,error,sigmaB,bdata) 
    

def rdf_analyzer(patient,cell_selection,immuneGroup,cell_centers,cellTypes,vdW_diam,blocks,plot_to,plot,antipair):
    """
    Analyze RDF's of different cell types. The last two categories in cellTypes are all and anti pair and measure the interaction
    of all types and the anti pairwise interaction of two types. If the antipairwise correlation has to be measured two cell types
    and 'antipair' have to be input in cell selection.
    IN: cell_selection - [n_cell_types]: Cell types under consideration
        blocks - 1-n_cells: Data is block averaged over 'blocks' points
        plot - True/False: output a plot of rdf
    """
    #decimals = 5 #limit precision in finding the same radius -> 
    cell_centers = cell_centers*0.39 #convert distances in microns
    allsameindex=[]; r_pairlist = []
    rdf_arr = np.zeros((2,blocks,len(cell_selection)))
    for i in range(0,len(cell_selection)):
        cell_forRdf = cell_selection[i]
        print('Analyzing RDF of: '+cellTypes[cell_forRdf])
        if(cell_forRdf == len(cellTypes)-2): #All cell types: indistinguishable particles
            cell_index = np.arange(0,len(immuneGroup))
            cell_positions = cell_centers[cell_index,:]
            boxdim = np.mean([max(cell_positions[:,0]),max(cell_positions[:,1])]) - np.mean([min(cell_positions[:,0]),min(cell_positions[:,1])])
            energy,r_list = ERList(cell_positions,boxdim)
            
        elif(cell_forRdf == len(cellTypes)-1): #Antipair correlation between first two types
            cell_index = allsameindex.astype(int) ;
            cell_positions = cell_centers[cell_index,:];
            boxdim = np.mean([max(cell_positions[:,0]),max(cell_positions[:,1])]) - np.mean([min(cell_positions[:,0]),min(cell_positions[:,1])])
            energy,r_alllist = ERList(cell_positions,boxdim)
            #r_alllist =  np.round(r_alllist,decimals)
            r_antipairlist = r_alllist
            r_list = np.setdiff1d(r_alllist,r_pairlist)
            for i in range(0,len(r_pairlist)): #All interactions - pairwise interactions = antipairwise interactions
                #sameindex = np.where(r_antipairlist == r_pairlist[i])[0][0]; #use round(), to find existing same value
                sameindex = np.argmin(np.abs(r_antipairlist-r_pairlist[i]));print(sameindex);print(min(np.abs(r_antipairlist-r_pairlist[i])))
                r_antipairlist = np.delete(r_antipairlist,sameindex); print(len(r_antipairlist))
            r_list = r_antipairlist
        else: #Pair correlations of same type
            cell_index = np.where(immuneGroup==cell_forRdf)[0]
            cell_positions = cell_centers[cell_index,:]; 
            allsameindex = np.concatenate([allsameindex,cell_index]) ;
            boxdim = np.mean([max(cell_positions[:,0]),max(cell_positions[:,1])]) - np.mean([min(cell_positions[:,0]),min(cell_positions[:,1])])
            energy,r_list = ERList(cell_positions,boxdim)
            #r_list = np.round(r_list,decimals)
            r_pairlist = np.concatenate([r_pairlist,r_list])
              
        #Bin the distances-radii
        binvec = np.arange(0,max(r_list),0.4)
        r_hist = np.histogram(r_list,binvec)
        binvec = 0.5*(binvec[1]-binvec[0]) + binvec[:-1] 
        #Normalize the rdf by avrg cell density in every circumference segment
        dr = binvec[1]-binvec[0] 
        rho = len(cell_index)/(boxdim**2) #average density = #cells/tissueSurface
        norm_coeff = 2*np.pi*binvec*dr*rho*(len(cell_index))/2
        norm_r_hist = (r_hist[0]/norm_coeff)
        #Block the data to denoise
        [blockvec,blocktt,error,sigmaB,bdata] = blocking(binvec,norm_r_hist,blocks)
        
        rdf_arr[0,0:len(blocktt),i] = blocktt
        rdf_arr[1,0:len(blocktt),i] = blockvec[0:len(blocktt)]
 
    if (plot == True):
        to = int(len(blocktt)*plot_to-1); 
        colors = plt.cm.jet(np.linspace(0,1,len(cellTypes)-2))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(0,len(cell_selection)): 
            if (cell_selection[i] == len(cellTypes)-1):
                ax.plot(rdf_arr[0,0:to,i],rdf_arr[1,0:to,i],
                         color='g',label=cellTypes[cell_selection[i]])
            elif (cell_selection[i] == len(cellTypes)-2):
                ax.plot(rdf_arr[0,0:to,i],rdf_arr[1,0:to,i],
                         color='k',label=cellTypes[cell_selection[i]])    
            else:    
                ax.plot(rdf_arr[0,0:to,i],rdf_arr[1,0:to,i],
                         color=colors[cell_selection[i]],label=cellTypes[cell_selection[i]])
        ax.set_xlabel(r'r' + f' [micrometer]'); ax.set_ylabel('g(r)')
        major_ticks = np.arange(0, blocktt[to], 10)
        ax.set_xticks(major_ticks)
        ax.grid(which='major', alpha=0.5)
        fig.suptitle('RDF - P_'+str(patient)); #plt.title('Scale = 1:0.39 Micrometer')  
        plt.legend(); plt.show()
        
    return rdf_arr   

def mean_rdf(rdf_data,from_to,plot):
    """
    Out of a composition of rdf's -> calculate the average rdf
    IN: rdf_data - [0-r_bins/1-Frequenzy,data,cell_selection,patient]: rdf data for cell types and patients
        from_to - [from,to]: selects a region (radius) over which the data is averaged
        plot - True/False: Plot the rdf g(r) in the selected region from - to
    """    
    blocks = len(rdf_data[0,:,0,0])
    n_patients = len(rdf_data[0,0,0,:])
    ffrom = int(blocks*from_to[0]-1); to = int(blocks*from_to[1]-1)
    rdf_mean = np.mean(rdf_data[1,ffrom:to,2,:],axis=1)
    r = rdf_data[0,ffrom:to,0,0]
    
    if (plot==True):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(r[ffrom:to],rdf_mean[ffrom:to],
                                 color='k',label='All Types')
        ax.set_xlabel(r'r' + f' [micrometer]'); ax.set_ylabel('<g(r)>')
        major_ticks = np.arange(r[ffrom-1], r[to-1], 10)
        ax.set_xticks(major_ticks)
        ax.grid(which='major', alpha=0.5)
        fig.suptitle(f'Average RDF considering N = {n_patients} tissue samples'); #Scale = 1pixel:0.39 Micrometer'  
        plt.legend(); plt.show()
    return r, rdf_mean 


def LJhypothesis(r,eps,s):
    """
    Hypothesis for the potential u(r) = -ln(g(r))
    IN: r - [r0,...rN] must be globally defind befor using it in chisquared
        eps, s are the free parameters to which chisquared optimizes
    """
    utild = 4*eps*((s/r)**(8) - (s/r)**4) # or 6,3
    return utild

def f_eval(xmg,ymg,function):
    """
    Function to evaluate a 2D function
    IN: xmg - [N,N]: meshgrid for x values
        ymg - [N,N]: meshgrid for y values
        function - function handle: function to evaluate in 2D
    """
    x_flat = np.ndarray.flatten(xmg)
    y_flat = np.ndarray.flatten(ymg)
    xy = np.array([x_flat,y_flat]).T
    
    
    length = len(xy)
    z = np.zeros(len(xy))
    for i in range(0,length):
        z[i] = function(xy[i,:])
    z = np.reshape(z,np.shape(xmg))    
    return z

def fLJ(r,e,s):
    F = -(16*e/r)*(2*(s/r)**8-(s/r)**4)
    return F

def potplot(r,potential,eps_opt,s_opt,chisquared,sigma,p):
    
    #Plot the hypothesis and the actual data# VLJ = 
    rr = np.linspace(min(r),max(r),200)
    plt.figure()
    plt.plot(rr,LJhypothesis(rr,eps_opt,s_opt),'--r',label=r'$V_{LJ}(r, e, s)$ Hypothesis')
    plt.plot(r,potential,label='-ln(g(r)) Data',c='k')
    plt.plot(rr,fLJ(rr,eps_opt,s_opt) ,label=r'$f(r, e, s)$ Force') #f(r) = -du(r)/dr
    plt.title('Potential - Force'); plt.xlabel('r [micrometer]'); plt.ylabel('u(r), f(r)')
    plt.legend(); plt.show()
    
    #%% Plot chisquared as function of free parameters eps & s:
    de = eps_opt*0.15; ds = s_opt*0.15
    ee = np.linspace(eps_opt-de,eps_opt+de,200) #loc parameter space
    ss = np.linspace(s_opt-ds,s_opt+ds,200)   #scale parameter space
    [emg,smg] = np.meshgrid(ee,ss)
    
    chisq_mg = f_eval(emg,smg,chisquared)
    
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(emg, smg, chisq_mg)
    plt.scatter(eps_opt,s_opt,chisquared(np.array([eps_opt,s_opt])),c='r')
    fig.colorbar(cp)
    fig.suptitle(r'$\chi^2$ for $V_{LJ}$ hypothesis')
    ax.set_title(r'$\sigma$ = '+f'{sigma}; p-value at minimum = {p:.3}'); ax.set_xlabel('e'); ax.set_ylabel('s'); 
    plt.show()
    
def antipairRlist(r_alllist,r_pairlist): 
    r_antipairlist = r_alllist
    for i in range(0,len(r_pairlist)):
        sameindex = np.where(r_antipairlist == r_pairlist[i]) ;print('index');print(np.shape(sameindex))
        r_antipairlist = np.delete(r_antipairlist,sameindex[0][0])
    #r_antipairlist = np.setdiff1d(r_alllist,r_pairlist)
    print('antipairlist'); print(len(r_antipairlist)+len(r_pairlist)-len(r_alllist))
    return r_antipairlist      

def clusterRestructure(cell_data,reset_cell,reset_immune,n_cell_clusters,n_immune_clusters,cell_col,immune_col,custom_col):  
    cell_data = np.append(cell_data,np.zeros((len(cell_data),1)),axis = 1)      
    for i in range(0,n_cell_clusters):
        idx = np.where(cell_data[:,cell_col] == i+1)
        cell_data[idx,custom_col] = reset_cell[i]
    
    for i in range(0,n_immune_clusters):
        idx = np.where(cell_data[:,immune_col] == i)
        cell_data[idx,custom_col] = reset_immune[i] 
    return cell_data

def clusterDepiction(cell_data,p_files,depiction):
    cell_col = 54; immune_col = 56; custom_col = 57
    n_cell_clusters = len(np.unique(cell_data[:,cell_col]))
    n_immune_clusters = len(np.unique(cell_data[:,immune_col]))
    if  (depiction == 'CellGroup'):
        immuneGroup = cell_data[p_files,cell_col] - 1  #excel labeling starts with 1-python 0
        cellTypes = np.array(['Undefined','Immune','Endothelial','Mesenchymal-like','Tumor','Kreatin-positive tumor'])   
    elif (depiction == 'ImmuneGroup'):
        immuneGroup = cell_data[p_files,56]   #immunge goups for patient in question
        cellTypes = np.array(['Tumor','Treg','CD4-T','CD8-T','CD3-T','NK','B','Neutrophils','Macrophages','DC','DC/Mono','Mono/Neu','Other Immune'])   
    elif depiction == 'Detailed':
        cellTypes = ['Undefined','Endothelial','Mesenchymal-like','Tumor & K+','Treg','CD4_T',
                          'CD8-T','CD3-T','NK','B','Neutrophils','Macrophages & Mono/Neu','DC & DC/Mono','Other Immune']
        reset_cell = [0,999,1,2,3,3]
        reset_immune = [3,4,5,6,7,8,9,10,11,12,12,11,13]
        cell_data = clusterRestructure(cell_data,reset_cell,reset_immune,n_cell_clusters,
                                       n_immune_clusters,cell_col,immune_col,custom_col)
        immuneGroup = cell_data[p_files,custom_col]
    elif depiction == 'Grained':
        cellTypes = ['Other','Endothelial','Mesenchymal-like','Tumor & K+','T-cells','NK','B',
                     'Granulocytes','DCs']
        reset_cell = [0,999,1,2,3,3]
        reset_immune = [3,4,4,4,4,5,6,7,7,8,8,7,0]
        cell_data = clusterRestructure(cell_data,reset_cell,reset_immune,n_cell_clusters,
                               n_immune_clusters,cell_col,immune_col,custom_col)
        immuneGroup = cell_data[p_files,custom_col]
    elif depiction == 'General':
        cellTypes = ['Other','Immune','Endothelial','Mesenchymal-like','Tumor']
        reset_cell = [0,1,2,3,4,4]
        reset_immune = [4,1,1,1,1,1,1,1,1,1,1,1,1]
        cell_data = clusterRestructure(cell_data,reset_cell,reset_immune,n_cell_clusters,
                               n_immune_clusters,cell_col,immune_col,custom_col)
        immuneGroup = cell_data[p_files,custom_col]
           
    tab20 = cm.get_cmap('tab20', 20)
    mycmap = tab20.colors[0:len(cellTypes),:3]
    n_cell_types = len(cellTypes)
    return immuneGroup,mycmap,cellTypes,n_cell_types

def pathology(cell_data,patient,depiction,plotArr,plotImg):
    """
    Creates a pathological image & numpy array of the tumor tissue pathology. 
    The .tiff file of the corresponding patient has to be included in the working directory
    IN: cell_data - Data of cell classification (Keren et al cell:2018)
        patient - int(1-41)
        depiction- "CellGroup" Depiction cell groups/ "ImmuneGroup" Depiction of immune groups
        plot - True/Fales: outputs a plot
    OUT: cell_centers -[n_cells,2] x-y position of cell (scale: 1pixel= 0.39micrometer)  
         immuneGroup - [ncells]
    """
    img = io.imread('p'+str(int(patient))+'_labeledcellData.tiff') #read grayscale values from .tiff image files 
    #out of all the patientsfind the indices for patient in question
    p_files = np.where(cell_data[:,0]==patient)[0] 

    immuneGroup,mycmap,cellTypes,n_cell_types = clusterDepiction(cell_data,p_files,depiction)
        
    cellLabelInImg = cell_data[p_files,1] #cellIDs for patient in question = grayscale expression
    n_cells= len(cellLabelInImg) #Number of distinct cells for every patient
    
    #create create cmap for plotImg
    cmap = plt.cm.gray
    norm = plt.Normalize(img.min(), img.max())
    rgba = cmap(norm(img))

    cell_centers = np.zeros((n_cells,2)) #preallocate for cells center array
    
    #color the cells according to immungeGroup (PlotImg)
    for i in range(0,n_cells): 
        cellLoc = np.where(img == cellLabelInImg[i]) #locate the cell according to grayscale/cell- ID
        x = cellLoc[0][:] #rows of cell
        y = cellLoc[1][:] #col of cell
        rgba[x, y, :3] = mycmap[int(immuneGroup[i])] #color the cell
        
        x_center = np.mean(x)
        y_center = np.mean(y)
        cell_centers[i,:] = [x_center,y_center]
 # PlotArr   
    if(plotArr == True):        
        tag = immuneGroup[0:n_cells]
        # create discrete cmap
        cmap, norm, bounds = discrete_cmap(n_cell_types)
        # setup the plot
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        # make the scatter
        scat = ax.scatter(cell_centers[:,0],cell_centers[:,1],c=tag,s=2.5,cmap=cmap,norm=norm)
        # create the colorbar
        cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
        cb.set_ticklabels(cellTypes)
        cb.set_label('Cell Types')
        fig.suptitle('Pathological image of tumor tissue. P_'+str(int(patient)))
        ax.set_title('Scale: 1:0.39 micrometer')
        plt.gca().invert_yaxis() #invert y axis to match plotImg
        #plt.gca().invert_xaxis() #invert x axis to match plotImg
        plt.show()
# PlotImg 
    if (plotImg == True) :   
        plt.figure()
        plt.imshow(rgba, interpolation='nearest')
        plt.suptitle('Pathological image of tumor tissue. P_'+str(int(patient)))
        plt.title('Scale: 1:0.39 micrometer')
        plt.show()
        
    return   cell_centers, immuneGroup, np.array(cellTypes) 

