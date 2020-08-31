#cython: language_level=3
# --compile-args="-O2"


cimport cython
import numpy as np

cimport numpy as cnp
from numpy cimport npy_intp, npy_longdouble
from libcpp cimport bool
from libc.math cimport exp


@cython.boundscheck(False)
cdef int calc_dE(npy_intp site,
                cnp.int32_t[::1] spins,
                cnp.int32_t[:, :] neighbors):    
    """Calculate e_jk: the energy change for spins[site] -> *= -1"""
    
    cdef:
        int site1 = spins[site]
        int dE
        npy_intp idx0 = neighbors[site, 0]
        npy_intp idx1 = neighbors[site, 1]
        npy_intp idx2 = neighbors[site, 2]
        npy_intp idx3 = neighbors[site, 3]
        npy_intp idx4 = neighbors[site, 4]
        npy_intp idx5 = neighbors[site, 5]
           
    
    dE = 2*site1*(spins[idx0] + spins[idx1]*spins[idx2] + 
                 spins[idx1]*spins[idx4] + spins[idx3] +
                 spins[idx4]*spins[idx5])
    
    return dE


@cython.cdivision(True)
cdef bint mc_choice(int dE, double T, npy_longdouble uni):
    """принимаем или не принимаем переворот спина?"""
    cdef double r
    r = exp(-dE/T)
    if dE <= 0:
        return True
    elif uni <= r:
        return True
    else:
        return False

    

@cython.boundscheck(False) 
cdef void step(cnp.int32_t[::1] spins, cnp.int32_t[:, :] neigh,
                double T, npy_intp site, npy_longdouble uni):
    """крутим 1 спин"""
        
    cdef int L2, dE
    
    L2 = spins.shape[0]
    
    dE = calc_dE(site, spins, neigh)   # if lattice s -> pass s,t
    if mc_choice(dE, T, uni):
        spins[site] *= -1
        

@cython.boundscheck(False)
def mc_step(cnp.int32_t[::1] spins,
            cnp.int32_t[:, :] neighbors,
            double T):
    """perform L*L flips for 1 MC step"""
    
    cdef npy_intp num_steps = spins.shape[0]
    cdef int _
    
    cdef cnp.ndarray[double,
                ndim=1,
                negative_indices=False,
                mode='c'] unis = np.random.uniform(size=num_steps)
    cdef cnp.ndarray[npy_intp,
                ndim=1,
                negative_indices=False,
                mode='c'] sites = np.random.randint(num_steps, size=num_steps)
    
    for _ in range(num_steps):
        step(spins, neighbors, T, sites[_], unis[_])
        
        

@cython.boundscheck(False)  
# @cython.cdivision(True)
cdef double calc_e_c(cnp.int32_t[::1] spins,
                  cnp.int32_t[:, :] neighbors):
    cdef npy_intp L2 = spins.shape[0]
    cdef int site,j,idx
    cdef int E = 0
    cdef double r
    cdef npy_intp idx0,idx1,idx2,idx3,idx4
    
    for site in range(L2):
        idx0 = neighbors[site, 0]
        idx1 = neighbors[site, 1]
        idx2 = neighbors[site, 2]
        idx3 = neighbors[site, 3]
        idx4 = neighbors[site, 4]
        E += spins[site]*(spins[idx0] + spins[idx3] + 
                spins[idx1]*spins[idx2] +
                spins[idx4]*spins[idx1])
    r = -E/L2/2
    return r

def calc_e(cnp.int32_t[::1] spins,
            cnp.int32_t[:, :] neighbors):
    cdef double E
    E = calc_e_c(spins, neighbors)
    return E


@cython.boundscheck(False)  
# @cython.cdivision(True)
cdef double calc_m_c(cnp.int32_t[::1] spins):
    cdef npy_intp L2 = spins.shape[0]
    cdef int site,j,idx
    cdef int M = 0
    cdef double r
    
    for site in range(L2):
        M += spins[site]
    r = M/L2
    return r

def calc_m(cnp.int32_t[::1] spins):
    cdef double M
    M = calc_m_c(spins)
    return M