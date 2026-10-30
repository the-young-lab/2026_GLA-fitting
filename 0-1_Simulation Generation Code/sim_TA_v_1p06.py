#!/usr/bin/env python3
###############################################################################
# Chris Rumble
# 3/31/23
#
# A series of functions for simulating TA data using a three-state model. This
# is an new version not compatible with the first iteration from last summer.
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

############################################################################
# define a class to carry all model parameters and perform the simulations #
############################################################################
class simulation:
    # this function runs when the object is created
    def __init__(self, widget, A, B, G, rxn, ins):       
        timer = time.time()
        
        self.widget = widget
        self.A   = A
        self.B   = B
        self.G   = G
        self.rxn = rxn
        self.ins = ins
        
        #########################################
        # common parameters for all simulations #
        #########################################
        # energy space
        self.nfreq  = int(200)
        self.wave    = np.linspace(200, 800, self.nfreq)  # in nm
        self.freq    = 1/self.wave*1e4                    # in kcm-1
        
        # time space for final spectra
        self.t_short = np.arange(-2, 2.00, 0.01)
        self.t_long  = np.logspace(np.log10(2), np.log10(5000), 100)
        self.t       = np.hstack((self.t_short, self.t_long))
        self.nt      = len(self.t)
        
        # time space for simulation
        self.t_full = np.arange(self.t.min(),
                                1.1*self.t.max(),
                                0.01)
        self.nt_full = len(self.t_full)
        
        # maximum deltaA in mOD
        self.abs_max = 40
        
        # perform the simulations
        self.three_state()
        self.gen_spec()
        
        self.widget.insert('end', '\n\nTotal elapsed time: %3.2f s' % (time.time() - timer))
        # print('Total elapsed time: %3.2f s' % (time.time() - timer))
        # print('')
        
    ################################
    # reversible three-state model #
    ################################
    def three_state(self):
        self.widget.insert('end', '\n\nGenerating kinetics...')
        #print('Generating kinetics...')
        timer = time.time()
        
        # unpack the model parameters, expected as time constants in ps
        A0  = 1
        kA  = 1/self.A[0]
        kB  = 1/self.B[0]
        kAB = 1/self.rxn[0]
        kBA = 1/self.rxn[1]
        
       # calculate the constants that show up in the model
        X     = kA + kAB
        Y     = kB + kBA
        Lp    = 0.5*(X + Y + np.sqrt((X-Y)**2 + 4*kAB*kBA))
        Lm    = 0.5*(X + Y - np.sqrt((X-Y)**2 + 4*kAB*kBA))
        
        # preallocate the populations
        self.pop = np.zeros((self.nt_full, 3))
        
        # population of A
        self.pop[:,0] = A0/(Lp - Lm)
        self.pop[:,0] = self.pop[:,0]*((Y  - Lm)*np.exp(-Lm*self.t_full) + 
                                       (Lp -  Y)*np.exp(-Lp*self.t_full))
        
        # population of B
        self.pop[:,1] = kAB*A0/(Lp - Lm)
        self.pop[:,1] = self.pop[:,1]*(np.exp(-Lm*self.t_full) - 
                                       np.exp(-Lp*self.t_full))
        
        # population of G
        self.pop[:,2] = (A0/(Lm - Lp))*((Y - Lm + kAB)*np.exp(-Lm*self.t_full) + 
                                        (Lp - Y - kAB)*np.exp(-Lp*self.t_full))
        
        # set the t<0 populations to 0
        self.pop[self.t_full<0, :] = 0
        
        self.pred = [1/Lp, 1/Lm]        
        
        self.widget.insert('end', '\n\nElapsed time: %3.2f s' % (time.time() - timer))
        #print('Elapsed time: %3.2f s' % (time.time() - timer))
        
    #####################
    # spectra generator #
    #####################
    def gen_spec(self):
        self.widget.insert('end', '\n\nGenerating spectra...')
        #print('Generating spectra...')
        timer = time.time()
        
        # unpack the parameters
        tau_mu   = self.A[1]
        tau_ga   = self.A[2]
        mu_A0    = self.A[3]
        mu_AI    = self.A[4]
        ga_A0    = self.A[5]
        ga_AI    = self.A[6]
        
        mu_B     = self.B[1]
        ga_B     = self.B[2]
        
        mu_G     = self.G[0]
        ga_G     = self.G[1]
        
        ga_noise = self.ins[0]
        ga_IRF   = self.ins[1]
        
        # pre-allocate some things
        self.comp     = list()
        self.spc_par  = list()

        ###############################################
        # Generate time-dependent spectral parameters #
        ###############################################
        # state A area, mean, and FWHM
        self.spc_par.append(np.zeros((self.nt_full, 3)))
        self.spc_par[0][:,0] = self.pop[:,0]
        self.spc_par[0][:,1] = (mu_A0 - mu_AI)*np.exp(-self.t_full/tau_mu) + mu_AI
        self.spc_par[0][:,2] = (ga_A0 - ga_AI)*np.exp(-self.t_full/tau_ga) + ga_AI

        # state B area, mean, and FWHM
        self.spc_par.append(np.zeros((self.nt_full, 3)))
        self.spc_par[1][:,0] = self.pop[:,1]
        self.spc_par[1][:,1] = np.ones(self.nt_full)*mu_B
        self.spc_par[1][:,2] = np.ones(self.nt_full)*ga_B

        # state G area, mean, and FWHM
        self.spc_par.append(np.zeros((self.nt_full, 3)))
        self.spc_par[2][:,0] = self.pop[:,2]
        self.spc_par[2][:,1] = np.ones(self.nt_full)*mu_G
        self.spc_par[2][:,2] = np.ones(self.nt_full)*ga_G

        ##########################################################
        # generate component spectra, make the full TA lineshape #
        ##########################################################
        self.spec = np.zeros((self.nfreq, self.nt_full))
        for i in range(3):
            self.comp.append(np.zeros([self.nfreq, self.nt_full]))
            for j in range(self.nt_full):
                self.comp[i][:,j] = gauss(self.spc_par[i][j,:], self.freq)
            self.spec = self.spec + self.comp[i]

        # convert from lineshapes into spectra
        for i in range(self.nt_full):
            self.spec[:,i] = self.spec[:,i]*self.freq

        # set maximum signal to 40 mOD
        self.spec = self.abs_max*self.spec/self.spec.max()
        
        self.widget.insert('end', '\n\nElapsed time: %3.2f s' % (time.time() - timer))
        #print('Elapsed time: %3.2f s' % (time.time() - timer))
        
        ##############################
        # convolve with Gaussian IRF #
        ##############################
        self.widget.insert('end', '\n\nConvoluting with IRF...')
        #print('Convoluting with IRF...')
        timer = time.time()
        
        self.irf = gauss([1, 0, ga_IRF], self.t_short)
        self.irf = self.irf/self.irf.max()

        for i in range(self.nfreq):
            self.spec[i,:] = np.convolve(self.spec[i,:], 
                                         self.irf, 
                                         mode='same')
        self.widget.insert('end', '\n\nElapsed time: %3.2f s' % (time.time() - timer))
        #print('Elapsed time: %3.2f s' % (time.time() - timer))
    
        ################################
        # interpolate from t_full to t #
        ################################
        self.widget.insert('end', '\n\nInterpolating...')
        #print('Interpolating...')
        timer = time.time()
        
        # populations
        dum = np.zeros((self.nt, 3))
        for i in range(3):
            dum[:,i] = np.interp(self.t, self.t_full, self.pop[:,i])
            dum[self.t<0,i] = 0
        self.pop = dum
            
        # simulation parameters
        for i in range(3):
            dum = np.zeros((self.nt, 3))
            for j in range(3):
                dum[:,j] = np.interp(self.t, 
                                     self.t_full, 
                                     self.spc_par[i][:,j])
                dum[self.t<0,j] = 0
            self.spc_par[i] = dum
            
        # components
        for i in range(3):
            dum = np.zeros((self.nfreq, self.nt))
            for j in range(self.nfreq):
                dum[j,:] = np.interp(self.t, 
                                     self.t_full, 
                                     self.comp[i][j,:])
            self.comp[i] = dum
            
        # spectra
        dum = np.zeros((self.nfreq, self.nt))
        for j in range(self.nfreq):
            dum[j,:] = np.interp(self.t, 
                                 self.t_full, 
                                 self.spec[j,:])
        self.spec = dum/dum.max()*self.abs_max
        
        self.widget.insert('end', '\n\nElapsed time: %3.2f s' % (time.time() - timer))
        #print('Elapsed time: %3.2f s' % (time.time() - timer))

        ###################################
        # add some uniform Gaussian noise #
        ###################################
        rng  = np.random.default_rng(seed=1337)
        self.spec = self.spec + rng.normal(0, 
                                           ga_noise, 
                                           size=np.shape(self.spec))
        
        self.widget.insert('end', '\n\nSpectral generation complete.')
        #print('Spectral generation complete.')
    
    ##############################
    # plot the simulation result #
    ##############################
    def plot_sim(self):
        ################
        # plot results #
        ################
        print('Making plots...')
        labels = [r'[A]', r'[B]', r'[G]']
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax = np.reshape(ax, np.size(ax))
        fig.set_size_inches([11, 8.5])
        fig.subplots_adjust(top=0.945,
                            bottom=0.1,
                            left=0.11,
                            right=0.955,
                            hspace=0.325,
                            wspace=0.28)        

        # population dynamics
        for i in range(3):
            ax[0].plot(self.t, self.pop[:,i], label=labels[i])
        ax[0].plot([-10, 1e5], [0, 0], 'k', alpha=0.5)
        ax[0].set_xlim([0.1, self.t[-1]])
        ax[0].set_xscale('symlog')
        ax[0].set_ylim([-self.pop.max()*1.5, self.pop.max()*1.5])
        ax[0].set_xlabel(r'Time / ps')
        ax[0].set_ylabel(r'Population')
        ax[0].legend(fontsize=12, ncol=3, loc='upper center')
        ax[0].set_title(r'\textbf{Population Dynamics}', fontsize=18)
        
        # sample spectra
        sel_spec = np.logspace(np.log10(1e-1), np.log10(self.t.max()), 10)
        ind_spec = np.zeros(len(sel_spec), dtype=int)
        spec_col = cm.get_cmap('brg', len(sel_spec))
        
        ax[1].plot([-10, 1e5], [0, 0], 'k', alpha=0.5)
        for i in range(len(sel_spec)):
            ind_spec[i] = int(np.abs(self.t-sel_spec[i]).argmin())
            ax[1].plot(self.wave, 
                       self.spec[:,ind_spec[i]], 
                       color=spec_col(i),
                       label=r'%3.1f ps' % self.t[ind_spec[i]])
            
        ax[1].plot([-10, 50], [0, 0], 'k')
        ax[1].set_xlabel(r'Wavelength / nm')
        ax[1].set_ylabel(r'$\Delta A$ / $10^{-3}$')
        
        if np.abs(self.spec.min()) < np.abs(self.spec.max()):
            ax[1].set_ylim([-self.spec.max()*1.1, self.spec.max()*1.1])
        else:
            ax[1].set_ylim([self.spec.min()*1.1, -self.spec.min()*1.1])
        ax[1].set_xlim([self.wave[0], self.wave[-1]])
        ax[1].legend(fontsize=10, ncol=2, loc='lower right')
        ax[1].set_title(r'\textbf{Individual Spectra}', fontsize=18)
        
        # sample time traces and IRF
        sel_trace = np.linspace(300, 600, 6, dtype=int)
        ind_trace = np.zeros(len(sel_trace), dtype=int)
        trace_col = cm.get_cmap('brg', len(sel_trace))

        ax[3].plot(self.t_short, self.irf*self.abs_max,
                   'gray', alpha=0.5, label='IRF')
        for i in range(len(sel_trace)):
            ind_trace[i] = int(np.abs(self.wave-sel_trace[i]).argmin())
            
            ax[3].plot(self.t, self.spec[ind_trace[i],:],
                       color=trace_col(i), 
                       label='%3d nm' % self.wave[ind_trace[i]])
            
        ax[3].set_xlim([0.1, self.t[-1]])
        ax[3].set_xscale('symlog') 
        ax[3].set_xlabel('Time / ps')
        ax[3].set_ylabel(r'$\Delta A$ / $10^{-3}$')
        ax[3].legend(fontsize=10)
        ax[3].set_title(r'\textbf{Sample Time Traces}', fontsize=18)

        # spectra as a contour plot
        if self.spec.max() > np.abs(self.spec.min()):
            vmin = -self.spec.max()
            vmax =  self.spec.max()
        else:
            vmin = -np.abs(self.spec.min())
            vmax =  np.abs(self.spec.min())
            
        ax[2].contourf(self.wave, self.t, self.spec.T, 16, 
                       cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax[2].set_ylim([0.1, self.t[-1]])
        ax[2].set_yscale('symlog')
        ax[2].set_ylabel('Time / ps')
        ax[2].set_xlabel(r'Wavelength / nm')
        ax[2].set_ylim([0.1, self.t[-1]])
        ax[2].tick_params(direction='out', which='both')
        ax[2].set_title(r'\textbf{Spectral Surface}', fontsize=18)

    ###################################
    # spit some text about the result #
    ###################################
    def text_result(self):
        self.widget.insert('end', '\n#########################################')
        self.widget.insert('end', '\nKinetic Parameters:')
        self.widget.insert('end', '\ntau_A  = %3.2f ps' % self.A[0])
        self.widget.insert('end', '\ntau_B  = %3.2f ps' % self.B[0])
        self.widget.insert('end', '\ntau_AB = %3.2f ps' % self.rxn[0])
        self.widget.insert('end', '\ntau_BA = %3.2f ps' % self.rxn[1])
        self.widget.insert('end', '\n\nPredicted time constants:')
        self.widget.insert('end', '\ntau_1 = %3.2f ps' % self.pred[0])
        self.widget.insert('end', '\ntau_2 = %3.2f ps' % self.pred[1])
        self.widget.insert('end', '\n\nIRF FWHM: %3.2f ps' % self.ins[1])
        self.widget.insert('end', '\n#########################################')
        # print(r'#########################################')
        # print(r'Kinetic Parameters:')
        # print(r'tau_A  = %3.2f ps' % self.A[0])
        # print(r'tau_B  = %3.2f ps' % self.B[0])
        # print(r'tau_AB = %3.2f ps' % self.rxn[0])
        # print(r'tau_BA = %3.2f ps' % self.rxn[1])
        # print('')
        # print('Predicted time constants:')
        # print('tau_1 = %3.2f ps' % self.pred[0])
        # print('tau_2 = %3.2f ps' % self.pred[1])
        # print('')
        # print('IRF FWHM: %3.2f ps' % self.ins[1])
        # print(r'#########################################')
        # print('')
        return self.pred[0], self.pred[1]
    ########################
    # save spectra results #
    ########################        
    def save_spec(self, out_name):
        write_dat(self.wave, self.t, self.spec, out_name + '.csv')
        return self.wave, self.t, self.spec

##########################
# other helper functions #
##########################
# define a gaussian function in terms of its height, mean, and FWHM
def gauss(par, x):
    I     = par[0]
    mean  = par[1]
    std   = par[2]/(2*np.sqrt(2*np.log(2)))
    
    gauss = np.exp(-(x-mean)**2/(2*std**2))
    gauss = (I/(par[2]*np.sqrt(2*np.pi)))*gauss
    return gauss

# save a set of two-dimensional data in ideal.dat format
def write_dat(freq, t, data, output_name):
    out = np.zeros([len(freq)+1, len(t)+1])
    out[1:, 1:] = data
    out[1:,0]   = freq
    out[0,1:]   = t
    
    np.savetxt(output_name, out, delimiter=',')

# read a set of two-dimensional data in ideal.dat format
def read_dat(filename):
    data = np.loadtxt(filename, delimiter=',')
    t    = data[0,1:]
    freq = data[1:,0]
    spec = data[1:, 1:]
    
    return t, freq, spec

