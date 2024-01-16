# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 10:39:56 2022

@author: sebja
"""

from MR_env import MR_env as Environment

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm

import copy

import pdb

from datetime import datetime

class ANN(nn.Module):

    def __init__(self, n_in, n_out, nNodes, nLayers, 
                 activation='silu', out_activation=None,
                 scale = 1):
        super(ANN, self).__init__()
        
        self.prop_in_to_h = nn.Linear(n_in, nNodes)

        self.prop_h_to_h = nn.ModuleList(
            [nn.Linear(nNodes, nNodes) for i in range(nLayers-1)])

        self.prop_h_to_out = nn.Linear(nNodes, n_out)
        
        if activation == 'silu':
            self.g = nn.SiLU()
        elif activation == 'relu':
            self.g = nn.ReLU()
        elif activation == 'sigmoid':
            self.g= torch.sigmoid()
        
        self.out_activation = out_activation
        self.scale = scale

    def forward(self, x):

        # input into  hidden layer
        h = self.g(self.prop_in_to_h(x))

        for prop in self.prop_h_to_h:
            h = self.g(prop(h))

        # hidden layer to output layer
        y = self.prop_h_to_out(h)

        if self.out_activation == 'tanh':
            y = torch.tanh(y)
            
        y = self.scale * y 

        return y


class DDPG():

    def __init__(self, env: Environment, I_max=10, 
                 gamma=0.99,  
                 n_nodes=36, n_layers=6, 
                 lr=1e-3, sched_step_size = 100,
                 name=""):

        self.env = env
        self.gamma = gamma
        self.I_max = I_max
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.name =  name
        self.sched_step_size = sched_step_size
        self.lr = lr
        
        self.__initialize_NNs__()
        
        self.S = []
        self.I = []
        self.q = []
        self.r = []
        self.epsilon = []
        
        self.Q_loss = []
        self.pi_loss = []
        
        
        
        
    def __initialize_NNs__(self):
        
        # policy approximation
        #
        # features = S, I
        #
        self.pi = {'net': ANN(n_in=2, 
                              n_out=1, 
                              nNodes=self.n_nodes, 
                              nLayers=self.n_layers,
                              out_activation='tanh',
                              scale=self.I_max)}
        
        self.pi['optimizer'], self.pi['scheduler'] = self.__get_optim_sched__(self.pi)        
        
        # Q - function approximation
        #
        # features = S, I, q
        #
        self.Q_main = {'net' : ANN(n_in=3, 
                                  n_out=1,
                                  nNodes=self.n_nodes, 
                                  nLayers=self.n_layers) }

        self.Q_main['optimizer'], self.Q_main['scheduler'] = self.__get_optim_sched__(self.Q_main)
        
        self.Q_target = copy.deepcopy(self.Q_main)
        
        
    def __get_optim_sched__(self, net):
        
        optimizer = optim.AdamW(net['net'].parameters(),
                                lr=self.lr)
                    
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.sched_step_size,
                                              gamma=0.99)
    
        return optimizer, scheduler
        
    def __stack_state__(self, S, I):

        return torch.cat((S.unsqueeze(-1)/self.env.S_0-1.0,
                          I.unsqueeze(-1)/self.I_max), axis=-1)
    
    def __grab_mini_batch__(self, mini_batch_size):
        
        t = torch.rand((mini_batch_size))*self.env.N
        # t[-int(mini_batch_size*0.05):] = self.env.N
        
        S, I = self.env.Randomize_Start(mini_batch_size)
        
        return t, S, I
    
    def Update_Q(self, n_iter = 10, mini_batch_size=256, epsilon=0.02):
        
        
        for i in range(n_iter):	
            
            _, S, I = self.__grab_mini_batch__(mini_batch_size)
            
            self.Q_main['optimizer'].zero_grad()
            
            # concatenate states
            X = self.__stack_state__(S, I)

            I_p = self.pi['net'](X).detach() * torch.exp(epsilon*torch.randn((mini_batch_size,1)))
            
            Q = self.Q_main['net']( torch.cat((X, I_p/self.I_max),axis=1) )
                
            # step in the environment
            S_p, I_p, r = self.env.step(0, S, I, I_p.reshape(-1))
            
            # compute the Q(S', a*)
            X_p = self.__stack_state__(S_p, I_p)

            # optimal policy at t+1
            I_pp = self.pi['net'](X_p).detach()
            
            # compute the target for Q
            target = r.reshape(-1,1) + self.gamma * self.Q_target['net'](torch.cat((X_p, I_pp/self.I_max),axis=1))
            
            loss = torch.mean((target.detach() - Q)**2)
            
            # compute the gradients
            loss.backward()

            # perform step using those gradients
            self.Q_main['optimizer'].step()                
            self.Q_main['scheduler'].step() 
            
            self.Q_loss.append(loss.item())
            
        self.Q_target = copy.deepcopy(self.Q_main)
        
    def Update_pi(self, n_iter = 10, mini_batch_size=256, epsilon=0.02):
        
        for i in range(n_iter):
            
            _, S, I = self.__grab_mini_batch__(mini_batch_size)
            
            self.pi['optimizer'].zero_grad()
            
            # concatenate states 
            X = self.__stack_state__(S, I)

            I_p = self.pi['net'](X)
            
            Q = self.Q_main['net']( torch.cat((X, I_p/self.I_max),axis=1) )
            
            loss = -torch.mean(Q)
                
            loss.backward()
            
            self.pi['optimizer'].step()
            self.pi['scheduler'].step()
            
            self.pi_loss.append(loss.item())
            
    def train(self, n_iter=1_000, 
              n_iter_Q=10, 
              n_iter_pi=5, 
              mini_batch_size=256, 
              n_plot=100):
        
        self.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"))

        C = 100
        D = 100
        
        if len(self.epsilon)==0:
            self.count=0
            
        for i in tqdm(range(n_iter)):
            
            epsilon = np.maximum(C/(D+self.count), 0.02)
            self.epsilon.append(epsilon)
            self.count += 1

             
            self.Update_Q(n_iter=n_iter_Q, 
                          mini_batch_size=mini_batch_size, 
                          epsilon=epsilon)
            
            self.Update_pi(n_iter=n_iter_pi, 
                           mini_batch_size=mini_batch_size, 
                           epsilon=epsilon)

            if np.mod(i+1,n_plot) == 0:
                
                self.loss_plots()
                self.run_strategy(1_000, name= datetime.now().strftime("%H_%M_%S"), N=100)
                self.plot_policy()
                # self.plot_policy(name=datetime.now().strftime("%H_%M_%S"))
                
    def moving_average(self, x, n):
        
        y = np.zeros(len(x))
        y_err = np.zeros(len(x))
        y[0] = np.nan
        y_err[0] = np.nan
        
        for i in range(1,len(x)):
            
            if i < n:
                y[i] = np.mean(x[:i])
                y_err[i] = np.std(x[:i])
            else:
                y[i] = np.mean(x[i-n:i])
                y_err[i] = np.std(x[i-n:i])
                
        return y, y_err                
            
    def loss_plots(self):
        
        def plot(x, label, show_band=True):

            mv, mv_err = self.moving_average(x, 100)
        
            if show_band:
                plt.fill_between(np.arange(len(mv)), mv-mv_err, mv+mv_err, alpha=0.2)
            plt.plot(mv, label=label, linewidth=1) 
            plt.legend()
            plt.ylabel('loss')
            plt.yscale('symlog')
        
        fig = plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plot(self.Q_loss, r'$Q$', show_band=False)
        
        plt.subplot(1,2,2)
        plot(self.pi_loss, r'$\pi$')
        
        plt.tight_layout()
        plt.show()
        
    def run_strategy(self, nsims=10_000, name="", N = None):
        
        if N is None:
            N = self.env.N
        
        S = torch.zeros((nsims, N+1)).float()
        I  = torch.zeros((nsims, N+1)).float()
        I_p = torch.zeros((nsims, N+1)).float()
        r = torch.zeros((nsims, N)).float()

        S0 = self.env.S_0
        I0 = 0

        S[:,0] = S0
        I[:,0] = 0
        
        ones = torch.ones(nsims)
        
        for t in range(N):

            X = self.__stack_state__(S[:,t], I[:,t])
            
            I_p[:,t] = self.pi['net'](X).reshape(-1)

            S[:,t+1], I[:,t+1], r[:,t] = \
                self.env.step(t*ones, S[:,t], I[:,t], I_p[:,t])
                
        S = S.detach().numpy()
        I  = I.detach().numpy()
        I_p = I_p.detach().numpy()
        r = r.detach().numpy()

        t = self.env.dt*np.arange(0, N+1)/self.env.T
        
        plt.figure(figsize=(5,5))
        n_paths = 3
        
        def plot(t, x, plt_i, title ):
            
            # print(x.shape)
            # pdb.set_trace()
            
            qtl= np.quantile(x, [0.05, 0.5, 0.95], axis=0)
            # print(qtl.shape)
            
            plt.subplot(2, 2, plt_i)
            
            plt.fill_between(t, qtl[0,:], qtl[2,:], alpha=0.5)
            plt.plot(t, qtl[1,:], color='k', linewidth=1)
            plt.plot(t, x[:n_paths, :].T, linewidth=1)
            
            # plt.xticks([0,0.5,1])
            plt.title(title)
            plt.xlabel(r"$t$")
            
        plot(t, (S-S[:,0].reshape(S.shape[0],-1)), 1, r"$S_t-S_0$" )
        plot(t, I, 2, r"$I_t$")
        plot(t[:-1], np.cumsum(r, axis=1), 3, r"$r_t$")

        plt.subplot(2,2, 4)
        plt.hist(np.sum(r,axis=1), bins=51)


        plt.tight_layout()
        
        plt.savefig("path_"  +self.name + "_" + name + ".pdf", format='pdf', bbox_inches='tight')
        plt.show()
        
        # zy0 = self.env.swap_price(zx[0,0], rx[0,0], ry[0,0])
        # plt.hist(zy[:,-1],bins=np.linspace(51780,51810,31), density=True, label='optimal')
        # qtl_levels = [0.05,0.5,0.95]
        # qtl = np.quantile(zy[:,-1],qtl_levels)
        # c=['r','b','g']
        # for i, q in enumerate(qtl):
        #     plt.axvline(qtl[i], linestyle='--', 
        #                 linewidth=2, 
        #                 color=c[i],
        #                 label=r'${0:0.2f}$'.format(qtl_levels[i]))
        # plt.axvline(zy0,linestyle='--',color='k', label='swap-all')
        # plt.xlabel(r'$z_T^y$')
        # plt.legend()
        # plt.savefig('ddqn_zy_T.pdf', format='pdf',bbox_inches='tight')
        # plt.show()
        
        # print(zy0, np.mean(zy[:,-1]>zy0))
        # print(qtl)        
        
        return t, S, I, I_p

    def plot_policy(self, name=""):
        
        NS = 101
        
        S = torch.linspace(self.env.S_0 - 3*self.env.inv_vol, 
                           self.env.S_0 + 3*self.env.inv_vol,
                           NS)
        NI = 51
        I = torch.linspace(-self.I_max, self.I_max, NI)
        
        Sm, Im = torch.meshgrid(S, I,indexing='ij')
        
        def plot(a, title):
            
            fig, ax = plt.subplots()
            plt.title("Inventory vs Price Heatmap for Time T")
            
            cs = plt.contourf(Sm.numpy(), Im.numpy(), a, 
                              levels=np.linspace(-self.I_max, self.I_max, 21),
                              cmap='RdBu')
            plt.axvline(self.env.S_0, linestyle='--', color='g')
            plt.axvline(self.env.S_0-2*self.env.inv_vol, linestyle='--', color='k')
            plt.axvline(self.env.S_0+2*self.env.inv_vol, linestyle='--', color='k')
            plt.axhline(0, linestyle='--', color='k')
            plt.axhline(self.I_max/2, linestyle='--', color='k')
            plt.axhline(-self.I_max/2, linestyle='--', color='k')
            ax.set_xlabel("Price")
            ax.set_ylabel("Inventory")
            ax.set_title(title)
            
            cbar = fig.colorbar(cs, ax=ax, shrink=0.9)
            cbar.set_ticks(np.linspace(-self.I_max, self.I_max, 11))
            cbar.ax.set_ylabel('Action')
                
            plt.tight_layout()
            plt.show()


        # X = torch.cat( ((Sm.unsqueeze(-1)/self.env.S_0-1.0), 
        #                 Im.unsqueeze(-1)/self.I_max), axis=-1)
        
        X = self.__stack_state__(Sm, Im)
        
        a = self.pi['net'](X).detach().squeeze()
        
        plot(a, r"")        
        
