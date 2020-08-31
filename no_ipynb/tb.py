#Uses python3

import numpy as np
import pandas as pd 
import time
import sys
import cy

#====================== all functions ======================
def coord(site):
    """get coordinate i of vector"""
    x = site // L
    y = site - x*L
    return (x,y)

def get(i):
    """fixin' boundary"""
    if i<0: return i
    else: return i % L
    
def get_neigh():
    """get neighbour's arr"""
    s = np.arange(L**2).reshape(L,L)
    nei = []
    for site in range(L*L):
        i,j = coord(site)
        nei += [s[get(i-1),get(j)],s[get(i),get(j+1)],s[get(i+1),get(j)],s[get(i),get(j-1)]]
    return np.array(nei, dtype=np.int32).reshape(L*L,4)

#################################################################
def gen_state():
    """generate random start state with lenght L*L and [-1,1] components"""
    state = np.array([np.random.choice([-1,1]) for _ in range(L*L)], dtype=np.int32)
    return state


def model_tb(L,T,N_avg=1,N_mc=1,Relax=1):
    """Моделируем АТ"""
    E, M = [], []
    s = gen_state()
    nei = get_neigh()
    
    #relax 10**3 times be4 AVG
    for __ in range(Relax):
        cy.mc_step(s, nei, T)
    #AVG every N_mc steps
    for _ in range(N_avg):
        for __ in range(N_mc):
            cy.mc_step(s, nei, T)
        E += [cy.calc_e(s,nei)]
        M += [cy.calc_m(s)]
    
    return E, M


def t_range(tc):
    t_ = np.array([0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3])
    t_low = np.round(-t_+tc, 7)                          #low
    t_high = np.round(t_+tc, 7)                      #high
    t = np.concatenate((t_low, t_high), axis=None)
    t.sort()
    return t

#################################################################

if __name__ == '__main__':

    global L
    L = 12

    seed = 1
    np.random.seed(seed)      # np.random.seed(int(sys.argv[1]))

    N_avg = 10**6
    N_mc = 10
    Relax = 10**6

    tc = 1/np.log(1+2**0.5)       # 2.2692
    t = t_range(tc)


    df_e, df_m = [pd.DataFrame() for i in range(2)]
    st = time.time()
    for ind,T in enumerate(t):
        e,m = model_tb(L,T,N_avg,N_mc,Relax)
        df_e.insert(ind,T,e, True)
        df_m.insert(ind,T,m, True)
    title = f"tb_L{L}_avg{N_avg}_mc{N_mc}_relax{Relax}"
    df_e.to_csv('export/e_'+title+'seed'+str(seed)+'.csv', index = None, header=True)
    df_m.to_csv('export/m_'+title+'seed'+str(seed)+'.csv', index = None, header=True)
    print('im done in ',time.time()-st)
