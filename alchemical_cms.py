#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from mendeleev.fetch import fetch_table
import numpy as np
from scipy.spatial import distance_matrix
import pandas as pd
from glob import glob
from joblib import Parallel, delayed, effective_n_jobs
from itertools import product


# In[3]:


# Find set of files
monomerfiles=glob('./dipeptides_coordinates/AA_AA_*/final.xyz')
co2files=glob('./dipeptides_co2_coordinates/AA_AA_*/final.xyz')


# In[3]:


# Name check
co2check=['_'.join(i.split('/')[-2].split('_')[:3]) for i in co2files]
moncheck=['_'.join(i.split('/')[-2].split('_')[:4]) for i in monomerfiles]


# In[4]:


# Intersection of the two sets
intersection=set(co2check)&set(moncheck)


# In[5]:


# Check for missing files and show them
mlen=len(monomerfiles)
clen=len(co2files)

print(f"Number of monomers is equivalent to monomers+CO2: {mlen==clen}")
print(f"Using the intersection of the sets of length: {len(intersection)}\n")

if mlen!=clen:
    print('Missing:')
    for idx,i in enumerate(set(moncheck)-set(co2check)):
        print(f"{idx}: {i}")


# In[6]:


def formatname(name):
    """
    parameters
    ----------
    name: str
        Name of complex
        
    returns
    -------
    system: str
        Monomer + CO2
    
    monomer: str
        Monomer
    """
    system=glob(f'./dipeptides_co2_coordinates/{name}_*/final.xyz')[0]
    monomer=f'./dipeptides_coordinates/{name}/final.xyz'
    return system,monomer


# In[7]:


# Threshold for covalent radii
thres=5e-1


# Useful data
ptable = fetch_table('elements').set_index('symbol')
# covradii=ptable['covalent_radius_pyykko']
# dblcovradii=ptable['covalent_radius_pyykko_double']
Z=ptable['atomic_number']


# In[8]:


def bond_check(atom_distance, minimum_length=0, maximum_length=1.5):
    """
    Thank you MolSSI: 
    https://education.molssi.org/python_scripting_cms/06-functions/index.html
    
    Check if a distance is a bond based on a minimum and maximum bond length"""
    
    if atom_distance >= minimum_length and atom_distance <= maximum_length:
        return True
    else:
        return False


# In[9]:


def gen_dist(xyzfile):
    """
    parameters
    ----------
    xyzfile: str
        Path to xyz
        
    returns
    -------
    atoms: numpy.ndarray
        Atom labels
        
    distmat: numpy.ndarray
        Distance matrix
        
    """
    structure=np.genfromtxt(xyzfile,skip_header=2,dtype=str)
    atoms=structure[:,0]
    xyz=structure[:,1:].astype(float)
    
    # Distance matrix
    distmat=distance_matrix(xyz,xyz)
    
    return atoms, distmat


# In[ ]:





# In[10]:


def genCM(aaname, xclosest=1, sort=True,penalty=None):
    """
    Generate (sorted) Coulomb matrix
    
    parameters
    ----------
    aaname: str
     Name of amino acid file
     
    xclosest: int
        Penalize x closest atoms (Default=1)
     
    sort: bool
        If true, return l2-norm row and column sorted Coulomb matrix
        
    penalty: float/int
        Value to penalize the values by
    
    returns
    -------
    M or sorted_CM: numpy.ndarray
        (sorted) Coulomb matrix
    
    """

    # Highly, highly dependent on CO2 being the last three atoms in the xyz file!!
    co2,mon=formatname(aaname)
    
    # Grab the distance matrix and indices
    co2atoms, co2distancematrix=gen_dist(co2)
    monatoms, mondistancematrix=gen_dist(mon)
    
    # CO2 info
    co2indx=np.arange(len(co2atoms))[-3:]
    numco2=len(co2indx)

    # Find indices that do not contain CO2
    nonco2idx=list(set(np.arange(len(co2atoms)))-set(co2indx))
    nonco2dist=co2distancematrix[co2indx[0],:][nonco2idx]
    nonco2lbl=co2atoms[nonco2idx]

    # Sort the indices that do not correspond to CO2
    findclose=np.array(sorted(zip(nonco2lbl,nonco2dist,nonco2idx),key=lambda x: x[1]),dtype=object)

    # Grab x closest and penalize them
    closest=findclose[0:xclosest,2].astype(int)

    # Generate Coulomb matrix
    M=np.zeros(np.shape(mondistancematrix))
    for idxi,i in enumerate(monatoms):
        for idxj,j in enumerate(monatoms):
            if idxi!=idxj:
                M[idxi,idxj]=(Z.loc[i]*Z.loc[j])/mondistancematrix[idxi,idxj]
            else:
                M[idxi,idxj]=0.5*Z.loc[i]**2.4
    
    if penalty!=None and all(co2atoms[:-3]==monatoms):
        # for i in closest:
        #     for j in closest:
        #         M[i,j]=penalty*M[i,j]
        if len(closest)!=1:
            M[closest,:][:,closest]=penalty*M[closest,:][:,closest]
        else:
            M[closest,closest]=penalty*M[closest,closest]

    if sort:
        # Create l2-norm sorted CM            
        row_idx=np.argsort(np.linalg.norm(M,axis=0))[::-1]
        col_idx=np.argsort(np.linalg.norm(M,axis=1))[::-1]

        sorted_CM=M[row_idx][:,col_idx]
        return sorted_CM
    else:
        return M        


# In[ ]:





# In[11]:


def pad(mat,maxs):
    """
    mat: numpy.ndarray
        Coulomb matrix to pad
    
    maxs: int
        Padding amount
    """
    npad=maxs-mat.shape[0]
    return np.pad(mat, [(0, npad), (0, npad)], mode='constant', constant_values=0).flatten()


# In[12]:


def genpaddedCMs(intersections,n_jobs=-1,xclosest=1, sort=True,penalty=None,**kwargs):
    """
    parameters
    ----------
    intersections: list
        Amino acid names that have both CO2+monomer and monomer files
        
    n_jobs: int
        Number of cores to run in parallel (Default=-1)
    
    *args: args
        CM arguments
        
    **kwargs: args
        Arguments for running joblib.Parallel
    
    """
    CMs=[genCM(i, xclosest,sort,penalty) for i in intersections]
    maxcm=max([i.shape[0] for i in CMs])

    C=np.array(Parallel(n_jobs=effective_n_jobs(n_jobs),**kwargs)(delayed(pad)(c,maxcm) for c in CMs))    
    
    return C


# In[13]:


def test1():
    '''
    Code test:
    Verify that our sorted Coulomb matrix is identical to the ones in Dscribe using padding!!
    '''
    # Import benchmark tools
    from dscribe.descriptors import CoulombMatrix
    from ase.build import molecule
    from ase.io import read
    mols=[]
    for k in sum([[i for i in monomerfiles if j=='_'.join(i.split('/')[-2].split('_')[:4])] for j in intersection],[]):
        mols.append(read(k))

    cm = CoulombMatrix(n_atoms_max=51,permutation='sorted_l2')
    CM=cm.create(mols)    

    # Ours
    C=genpaddedCMs(intersection,xclosest=1, sort=True,penalty=None)

    return np.isclose(C,CM).all()    



def test2():
    '''
    Code test:
    Verify that our sorted-penalized Coulomb matrix is not identical to the ones in Dscribe using padding!!
    '''
    # Import benchmark tools
    from dscribe.descriptors import CoulombMatrix
    from ase.build import molecule
    from ase.io import read
    mols=[]
    for k in sum([[i for i in monomerfiles if j=='_'.join(i.split('/')[-2].split('_')[:4])] for j in intersection],[]):
        mols.append(read(k))

    cm = CoulombMatrix(n_atoms_max=51,permutation='sorted_l2')
    CM=cm.create(mols)    

    # Ours
    C=genpaddedCMs(intersection,xclosest=1, sort=True,penalty=100)

    return np.isclose(C,CM).all()    


# In[ ]:





# In[14]:


if __name__=='__main__':
    if test1():
        print("ALL CMs PASS 1!")
    if test2()==False:
        print("ALL CMs PASS 2!")        


# In[ ]:




