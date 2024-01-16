# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

from MR_env import MR_env
from DDPG import DDPG

#%%
env = MR_env(S_0=1, kappa=1, sigma=0.2, theta=1,
             dt=0.25, T = int(20), 
             I_max=10, lambd=1)

ddpg = DDPG(env, I_max = 10,
            gamma = 0.999, 
            lr=1e-3,
            name="test" )
 
#%%    
ddpg.train(n_plot=200, n_iter_Q=5, n_iter_pi=5)