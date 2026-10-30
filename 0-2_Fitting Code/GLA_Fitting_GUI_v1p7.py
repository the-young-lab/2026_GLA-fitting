# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:00:24 2023

@author: Robert Hamburger
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import Tk, filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from scipy.optimize import least_squares
import sys

sys.path.insert(0, r'C:\Users\rhamb\Documents\Young Lab\Programming\Library')

import Function_Library_RH_v1p11 as rhlib
 
############################################# Fitter ####################################################

def GLA_fit(TD, Spec, Num_Exp, Use_Inf, Guess_Set, LowerB_Set, UpperB_Set):
    
    # Guess Construction    
    lifetimes = [Guess_Set[2], Guess_Set[3], Guess_Set[4], Guess_Set[5], Guess_Set[6], Guess_Set[7]]
    guess = np.zeros(Num_Exp + 2, dtype=float)

    for i in range(Num_Exp + 2):
        if i == 0:
            guess[i] = Guess_Set[0] 
        elif i == 1:
            guess[i] = Guess_Set[1]
        elif i > 1:
            guess[i] = lifetimes[i-2]
    
    # Bounds Construction
    l_bounds = [LowerB_Set[2], LowerB_Set[3], LowerB_Set[4], LowerB_Set[5], LowerB_Set[6], LowerB_Set[7]]
    u_bounds = [UpperB_Set[2], UpperB_Set[3], UpperB_Set[4], UpperB_Set[5], UpperB_Set[6], UpperB_Set[7]]

    bounds_v = []

    for j in range(2):
        empty = []
        for i in range(Num_Exp + 2):
            if j == 0 and i == 0:
                empty.append(LowerB_Set[0])
            elif j == 0 and i == 1:
                empty.append(LowerB_Set[1])
            elif j == 0 and i > 1:
                empty.append(l_bounds[i-2])
            elif j == 1 and i == 0:
                empty.append(UpperB_Set[0])
            elif j == 1 and i == 1:
                empty.append(UpperB_Set[1])
            elif j == 1 and i > 1:
                empty.append(u_bounds[i-2])
        bounds_v.append(empty)      
    
    # Data Fitting
    if Use_Inf in ['Yes']:        
        result = least_squares(rhlib.GLA_residuals, guess, bounds=bounds_v, args=(rhlib.GLA_model_i, TD, Spec))
        fit_par = result.x
        fit, a_mat, t_mat = rhlib.GLA_model_i(fit_par, TD, Spec)
    else:
        result  = least_squares(rhlib.GLA_residuals, guess, bounds=bounds_v, args=(rhlib.GLA_model, TD, Spec))
        fit_par = result.x
        fit, a_mat, t_mat = rhlib.GLA_model(fit_par, TD, Spec)
    
    # Residual Construction and Rating
    resid = Spec - fit    
    rcv = rhlib.matrixRSE(resid)
    
    #EADS              
    B      = np.zeros((Num_Exp, Num_Exp));
    B[:,0] = 1             
    for i in np.arange(2, Num_Exp+1, 1, dtype=int):
        for j in np.arange(i, Num_Exp+1, 1, dtype=int):
            B[j-1,i-1] = 1
            for k in np.arange(1, j, 1, dtype=int):
                B[j-1, i-1] = ((1/fit_par[k-1] - 1/fit_par[j-1])/
                               (1/fit_par[k-1]))*B[j-1, i-1] 

    # apply the B matrix to the DADS
    EADS = np.zeros(np.shape(a_mat))
    for i in range(Num_Exp):
        for j in range(Num_Exp):
            EADS[i,:] = EADS[i,:] + a_mat[j,:]*B[j,i]
    
    
    # Data Prep
    lts = np.zeros(Num_Exp + 1, dtype=float)

    for i in range(len(lts)):
        if i == 0:
            lts[i] = 100.0
        elif i > 0:
            lts[i] = fit_par[i+1]

    if Use_Inf in ['Yes']:
        dads_a = np.copy(a_mat.T)
    else:    
        dads_a = np.copy(a_mat.T)
        dads_a = np.insert(dads_a, 0, 0, axis=1)

    return fit, fit_par, dads_a, lts, rcv, resid, EADS
    
############################################# Fitter ####################################################


class Application(tk.Frame):
    def __init__(self, master=None):               
        tk.Frame.__init__(self, master)
        self.createframes()        
        self.createWidgets()
        self.master.geometry('500x500')


    def createframes(self):
        # Master Frame adjust with screen changes        
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        
        # Canvas window to hold everything so scroll bars can be added
        self.canvas_window = tk.Canvas(self.master)
        self.canvas_window.grid(row=0,column=0,sticky='nsew')
        self.canvas_window.configure(width=500,height=500)               
        
        # Scroll bars and scroll bar configuration
        self.scroll_y = ttk.Scrollbar(self.master, orient='vertical', command=self.canvas_window.yview)
        self.scroll_x = ttk.Scrollbar(self.master, orient='horizontal', command=self.canvas_window.xview)
        self.scroll_y.grid(row=0,column=1, sticky='ns')
        self.scroll_x.grid(row=1,column=0, sticky='ew')
        self.canvas_window.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
                
        # window frame in canvas for all main widgets
        self.window_frame = tk.Frame(master=self.canvas_window,)        
        
        # Sets window to dynamic
        self.window_frame.bind("<Configure>",lambda e: self.canvas_window.configure(scrollregion=self.canvas_window.bbox("all")))
        
        # Add window
        self.canvas_window.create_window((0,0), window=self.window_frame, anchor='nw')
        
        # Tool Bar Frame
        self.tool_frame = tk.Frame(master=self.window_frame)
        self.tool_frame.grid(row=0,column=0, columnspan=2, sticky = 'nw')
        
        # Main Fitting Stuff Frame
        self.main_frame = tk.Frame(master=self.window_frame)
        self.main_frame.grid(row=1,column=0, sticky ='nw')
            
        # Main Fitting Stuff Frame
        self.plot_frame = tk.Frame(master=self.window_frame)
        self.plot_frame.grid(row=1,column=1, sticky ='nw')
        
        # Lifetime Value Frame 
        self.lifetime_frame = tk.LabelFrame(self.main_frame, text="Model Lifetimes", labelanchor='n')
        self.lifetime_frame.grid(row=0, column=0, columnspan=10, padx=5, pady=5)

        # IRF and Bounds + Time Zero and Bounds Frame 
        self.irftz_frame = tk.LabelFrame(self.main_frame, text="IRF and Time Zero", labelanchor='n')
        self.irftz_frame.grid(row=1, column=0, columnspan=10, padx=5, pady=5)

        # Bounds Frame
        self.bounds_frame = tk.LabelFrame(self.main_frame, text="Bounds Values", labelanchor='n')
        self.bounds_frame.grid(row=2, column=0, columnspan=10, padx=5, pady=5)

        # Misc Frame
        self.misc_frame = tk.LabelFrame(self.main_frame, text="Misc. Parameters", labelanchor='n')
        self.misc_frame.grid(row=3, column=0, columnspan=10, padx=5, pady=5)
        
        # Button Frame
        self.button_frame = tk.LabelFrame(self.main_frame, text="", labelanchor='n')
        self.button_frame.grid(row=4, column=0, columnspan=10, padx=5, pady=5)        
        
        # Text Box
        self.load_box = tk.Text(self.main_frame, height=5, width=42, bg='white')
        self.load_box.grid(row=5, column=0, columnspan=10, padx=5, pady=5)
        
        # Text Box
        self.info_box = tk.Text(self.main_frame, height=10, width=42, bg='white')
        self.info_box.grid(row=6, column=0, columnspan=10, padx=5, pady=5)
  
        
    def createWidgets(self):        
        self.fig, self.ax = plt.subplots(nrows=2,ncols=3)        
        self.fig.set_size_inches([15,10])
        self.fig.subplots_adjust(top=0.926,bottom=0.058,left=0.061,right=0.967,hspace=0.205,wspace=0.232)        
        self.canvas_plot=FigureCanvasTkAgg(self.fig,master=self.plot_frame)
        
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky='nw')        
        
        for i in range(2):
            for j in range(3):
                self.ax[i,j].format_coord = lambda x, y: ""
        
        icmap = 'bone_r'
        sm = plt.cm.ScalarMappable(cmap=icmap)
        
        sm.format_coord = lambda x,y: ""
        
        self.cbar1 = self.fig.colorbar(sm, ax=self.ax[0,0])
        self.cbar2 = self.fig.colorbar(sm, ax=self.ax[0,1])
        self.cbar3 = self.fig.colorbar(sm, ax=self.ax[0,2])
        
        self.cbar1.format_coord = lambda x,y: ""
        self.cbar2.format_coord = lambda x,y: ""
        self.cbar3.format_coord = lambda x,y: ""     
        
        self.toolbar=NavigationToolbar2Tk(self.canvas_plot, self.tool_frame)
        self.canvas_plot.draw()

        # Number of Exponentials Label/Entry Widget
        self.num_exp_label = tk.Label(self.lifetime_frame, text="Number of Exponentials")
        self.num_exp_label.grid(row=0, column=0)

        self.num_exp_combobox = ttk.Combobox(self.lifetime_frame, state="readonly", values=[1,2,3,4,5,6], width=5)
        self.num_exp_combobox.grid(row=1, column=0)
        self.num_exp_combobox.current(0)

        self.use_inf_label = tk.Label(self.lifetime_frame, text="Use Unresolved")
        self.use_inf_label.grid(row=0, column=1)

        self.use_inf_status = tk.StringVar(value="No")
        self.use_inf_box = tk.Checkbutton(self.lifetime_frame, variable=self.use_inf_status, onvalue="Yes", offvalue="No")
        self.use_inf_box.grid(row=1, column=1)

        self.guess_frame = tk.LabelFrame(self.lifetime_frame, text="Guess Values", labelanchor='n')
        self.guess_frame.grid(row=2, column=0, columnspan=2)
                        
        # Exponential Row 1
        self.fix1_label = tk.Label(self.guess_frame, text="Fix")
        self.fix1_label.grid(row=0, column=2)
        
        self.fix2_label = tk.Label(self.guess_frame, text="Fix")
        self.fix2_label.grid(row=0, column=5)
        
        self.exp1_label = tk.Label(self.guess_frame, text="Lifetime 1")
        self.exp1_label.grid(row=1, column=0)
        self.exp1_entry = tk.Entry(self.guess_frame, width=10)
        self.exp1_entry.grid(row=1, column=1)
        self.exp1_entry.insert(0, "1")

        self.fix_exp1_status = tk.StringVar(value="No")
        self.fix_exp1_box = tk.Checkbutton(self.guess_frame, variable=self.fix_exp1_status, onvalue="Yes", offvalue="No")
        self.fix_exp1_box.grid(row=1, column=2) 

        self.exp2_label = tk.Label(self.guess_frame, text="Lifetime 2")
        self.exp2_label.grid(row=1, column=3)
        self.exp2_entry = tk.Entry(self.guess_frame, width=10)
        self.exp2_entry.grid(row=1, column=4)
        self.exp2_entry.insert(0, "1")

        self.fix_exp2_status = tk.StringVar(value="No")
        self.fix_exp2_box = tk.Checkbutton(self.guess_frame, variable=self.fix_exp2_status, onvalue="Yes", offvalue="No")
        self.fix_exp2_box.grid(row=1, column=5) 

        self.exp3_label = tk.Label(self.guess_frame, text="Lifetime 3")
        self.exp3_label.grid(row=2, column=0)
        self.exp3_entry = tk.Entry(self.guess_frame, width=10)
        self.exp3_entry.grid(row=2, column=1)
        self.exp3_entry.insert(0, "1")

        self.fix_exp3_status = tk.StringVar(value="No")
        self.fix_exp3_box = tk.Checkbutton(self.guess_frame, variable=self.fix_exp3_status, onvalue="Yes", offvalue="No")
        self.fix_exp3_box.grid(row=2, column=2) 

        # Exponential Row 2
        self.exp4_label = tk.Label(self.guess_frame, text="Lifetime 4")
        self.exp4_label.grid(row=2, column=3)
        self.exp4_entry = tk.Entry(self.guess_frame, width=10)
        self.exp4_entry.grid(row=2, column=4)
        self.exp4_entry.insert(0, "1")

        self.fix_exp4_status = tk.StringVar(value="No")
        self.fix_exp4_box = tk.Checkbutton(self.guess_frame, variable=self.fix_exp4_status, onvalue="Yes", offvalue="No")
        self.fix_exp4_box.grid(row=2, column=5) 

        self.exp5_label = tk.Label(self.guess_frame, text="Lifetime 5")
        self.exp5_label.grid(row=3, column=0)
        self.exp5_entry = tk.Entry(self.guess_frame, width=10)
        self.exp5_entry.grid(row=3, column=1)
        self.exp5_entry.insert(0, "1")

        self.fix_exp5_status = tk.StringVar(value="No")
        self.fix_exp5_box = tk.Checkbutton(self.guess_frame, variable=self.fix_exp5_status, onvalue="Yes", offvalue="No")
        self.fix_exp5_box.grid(row=3, column=2) 

        self.exp6_label = tk.Label(self.guess_frame, text="Lifetime 6")
        self.exp6_label.grid(row=3, column=3)
        self.exp6_entry = tk.Entry(self.guess_frame, width=10)
        self.exp6_entry.grid(row=3, column=4)        
        self.exp6_entry.insert(0, "1")

        self.fix_exp6_status = tk.StringVar(value="No")
        self.fix_exp6_box = tk.Checkbutton(self.guess_frame, variable=self.fix_exp6_status, onvalue="Yes", offvalue="No")
        self.fix_exp6_box.grid(row=3, column=5) 

        # Items in IRF and Time Zero Frame
        self.tz_label = tk.Label(self.irftz_frame, text="Time Zero Guess")
        self.tz_label.grid(row=0, column=0)
        self.tz_entry = tk.Entry(self.irftz_frame, width=10)
        self.tz_entry.grid(row=1, column=0)
        self.tz_entry.insert(0, "0")

        self.irf_label = tk.Label(self.irftz_frame, text="IRF Guess")
        self.irf_label.grid(row=0, column=1)
        self.irf_entry = tk.Entry(self.irftz_frame, width=10)
        self.irf_entry.grid(row=1, column=1)
        self.irf_entry.insert(0, "0.25")

        self.tzb_frame = tk.LabelFrame(self.irftz_frame, text="Time Zero Bounds", labelanchor='n')
        self.tzb_frame.grid(row=2, column=0)

        self.irfb_frame = tk.LabelFrame(self.irftz_frame, text="IRF Bounds", labelanchor='n')
        self.irfb_frame.grid(row=2, column=1)

        self.tzlb_label = tk.Label(self.tzb_frame, text="Lower Bound")
        self.tzlb_label.grid(row=0, column=0)
        self.tzub_label = tk.Label(self.tzb_frame, text="Upper Bound")
        self.tzub_label.grid(row=1, column=0)

        self.tzlb_entry = tk.Entry(self.tzb_frame, width=12)
        self.tzlb_entry.grid(row=0, column=1)
        self.tzub_entry = tk.Entry(self.tzb_frame, width=12)
        self.tzub_entry.grid(row=1, column=1)
        self.tzlb_entry.insert(0, "-5")
        self.tzub_entry.insert(0, "5")

        self.irflb_label = tk.Label(self.irfb_frame, text="Lower Bound")
        self.irflb_label.grid(row=0, column=0)
        self.irfub_label = tk.Label(self.irfb_frame, text="Upper Bound")
        self.irfub_label.grid(row=1, column=0)

        self.irflb_entry = tk.Entry(self.irfb_frame, width=12)
        self.irflb_entry.grid(row=0, column=1)
        self.irfub_entry = tk.Entry(self.irfb_frame, width=12)
        self.irfub_entry.grid(row=1, column=1)
        self.irflb_entry.insert(0, "0")
        self.irfub_entry.insert(0, "0.5")

        # Items in Bounds Frame
        # Lifetime Bounds Frame
        self.bounds1_frame = tk.LabelFrame(self.bounds_frame, text="Lifetime 1 Bounds", labelanchor='n')
        self.bounds1_frame.grid(row=1, column=0)

        self.bounds2_frame = tk.LabelFrame(self.bounds_frame, text="Lifetime 2 Bounds", labelanchor='n')
        self.bounds2_frame.grid(row=1, column=1)

        self.bounds3_frame = tk.LabelFrame(self.bounds_frame, text="Lifetime 3 Bounds", labelanchor='n')
        self.bounds3_frame.grid(row=2, column=0)

        self.bounds4_frame = tk.LabelFrame(self.bounds_frame, text="Lifetime 4 Bounds", labelanchor='n')
        self.bounds4_frame.grid(row=2, column=1)

        self.bounds5_frame = tk.LabelFrame(self.bounds_frame, text="Lifetime 5 Bounds", labelanchor='n')
        self.bounds5_frame.grid(row=3, column=0)

        self.bounds6_frame = tk.LabelFrame(self.bounds_frame, text="Lifetime 6 Bounds", labelanchor='n')
        self.bounds6_frame.grid(row=3, column=1)

        # Bounds Entry Row 1
        self.exp1lb_label = tk.Label(self.bounds1_frame, text="Lower Bound")
        self.exp1lb_label.grid(row=0, column=0)
        self.exp1ub_label = tk.Label(self.bounds1_frame, text="Upper Bound")
        self.exp1ub_label.grid(row=1, column=0)

        self.exp1lb_entry = tk.Entry(self.bounds1_frame, width=12)
        self.exp1lb_entry.grid(row=0, column=1)
        self.exp1ub_entry = tk.Entry(self.bounds1_frame, width=12)
        self.exp1ub_entry.grid(row=1, column=1)
        self.exp1lb_entry.insert(0, "0")
        self.exp1ub_entry.insert(0, "inf")

        self.exp2lb_label = tk.Label(self.bounds2_frame, text="Lower Bound")
        self.exp2lb_label.grid(row=0, column=0)
        self.exp2ub_label = tk.Label(self.bounds2_frame, text="Upper Bound")
        self.exp2ub_label.grid(row=1, column=0)

        self.exp2lb_entry = tk.Entry(self.bounds2_frame, width=12)
        self.exp2lb_entry.grid(row=0, column=1)
        self.exp2ub_entry = tk.Entry(self.bounds2_frame, width=12)
        self.exp2ub_entry.grid(row=1, column=1)
        self.exp2lb_entry.insert(0, "0")
        self.exp2ub_entry.insert(0, "inf")

        # Bounds Entry Row 2
        self.exp3lb_label = tk.Label(self.bounds3_frame, text="Lower Bound")
        self.exp3lb_label.grid(row=0, column=0)
        self.exp3ub_label = tk.Label(self.bounds3_frame, text="Upper Bound")
        self.exp3ub_label.grid(row=1, column=0)

        self.exp3lb_entry = tk.Entry(self.bounds3_frame, width=12)
        self.exp3lb_entry.grid(row=0, column=1)
        self.exp3ub_entry = tk.Entry(self.bounds3_frame, width=12)
        self.exp3ub_entry.grid(row=1, column=1)
        self.exp3lb_entry.insert(0, "0")
        self.exp3ub_entry.insert(0, "inf")
        
        self.exp4lb_label = tk.Label(self.bounds4_frame, text="Lower Bound")
        self.exp4lb_label.grid(row=0, column=0)
        self.exp4ub_label = tk.Label(self.bounds4_frame, text="Upper Bound")
        self.exp4ub_label.grid(row=1, column=0)

        self.exp4lb_entry = tk.Entry(self.bounds4_frame, width=12)
        self.exp4lb_entry.grid(row=0, column=1)
        self.exp4ub_entry = tk.Entry(self.bounds4_frame, width=12)
        self.exp4ub_entry.grid(row=1, column=1)
        self.exp4lb_entry.insert(0, "0")
        self.exp4ub_entry.insert(0, "inf")
        
        # Bounds Entry Row 3
        self.exp5lb_label = tk.Label(self.bounds5_frame, text="Lower Bound")
        self.exp5lb_label.grid(row=0, column=0)
        self.exp5ub_label = tk.Label(self.bounds5_frame, text="Upper Bound")
        self.exp5ub_label.grid(row=1, column=0)

        self.exp5lb_entry = tk.Entry(self.bounds5_frame, width=12)
        self.exp5lb_entry.grid(row=0, column=1)
        self.exp5ub_entry = tk.Entry(self.bounds5_frame, width=12)
        self.exp5ub_entry.grid(row=1, column=1)
        self.exp5lb_entry.insert(0, "0")
        self.exp5ub_entry.insert(0, "inf")

        self.exp6lb_label = tk.Label(self.bounds6_frame, text="Lower Bound")
        self.exp6lb_label.grid(row=0, column=0)
        self.exp6ub_label = tk.Label(self.bounds6_frame, text="Upper Bound")
        self.exp6ub_label.grid(row=1, column=0)

        self.exp6lb_entry = tk.Entry(self.bounds6_frame, width=12)
        self.exp6lb_entry.grid(row=0, column=1)
        self.exp6ub_entry = tk.Entry(self.bounds6_frame, width=12)
        self.exp6ub_entry.grid(row=1, column=1)
        self.exp6lb_entry.insert(0, "0")
        self.exp6ub_entry.insert(0, "inf")

        # Misc Frame Items        

        # Plot File Type
        self.pfile_type_label = tk.Label(self.misc_frame, text="Plot File Type")
        self.pfile_type_label.grid(row=0, column=0)

        self.pfile_type_combobox = ttk.Combobox(self.misc_frame, state="readonly", values=[".PNG",".SVG"], width=5)
        self.pfile_type_combobox.grid(row=1, column=0)
        self.pfile_type_combobox.current(0) 

        # Contour Plot Style
        self.csfile_type_label = tk.Label(self.misc_frame, text="Contour Style")
        self.csfile_type_label.grid(row=2, column=0)

        self.csfile_type_combobox = ttk.Combobox(self.misc_frame, state="readonly", values=["Fill","Line"], width=5)
        self.csfile_type_combobox.grid(row=3, column=0)
        self.csfile_type_combobox.current(0) 
        
        # Time Unit
        self.tunit_label = tk.Label(self.misc_frame, text="Time Units")
        self.tunit_label.grid(row=0, column=1)

        self.tunit_combobox = ttk.Combobox(self.misc_frame, state="readonly", values=["fs","ps","ns","us"], width=3)
        self.tunit_combobox.grid(row=1, column=1)
        self.tunit_combobox.current(1)

        # Energy Unit
        self.eunit_label = tk.Label(self.misc_frame, text="Energy Units")
        self.eunit_label.grid(row=2, column=1)

        self.eunit_combobox = ttk.Combobox(self.misc_frame, state="readonly", values=["nm","eV"], width=3)
        self.eunit_combobox.grid(row=3, column=1)
        self.eunit_combobox.current(0)

        # Plot Start Time
        self.p_start_label = tk.Label(self.misc_frame, text="Plot Start")
        self.p_start_label.grid(row=0, column=2)

        self.p_start_entry = tk.Entry(self.misc_frame, width=10)
        self.p_start_entry.grid(row=1, column=2)
        self.p_start_entry.insert(0, "-1.00")

        # Spectra Start Time
        self.s_start_label = tk.Label(self.misc_frame, text="Spectra Start")
        self.s_start_label.grid(row=2, column=2)

        self.s_start_entry = tk.Entry(self.misc_frame, width=10)
        self.s_start_entry.grid(row=3, column=2)
        self.s_start_entry.insert(0, "0.00")

        # DADS Colormap selection
        self.dcmap_label = tk.Label(self.misc_frame, text="DADS Colormap")
        self.dcmap_label.grid(row=0, column=3)

        self.dcmap_combobox = ttk.Combobox(self.misc_frame, state="readonly", values=["Custom","Tab10"], width=10)
        self.dcmap_combobox.grid(row=1, column=3)
        self.dcmap_combobox.current(1)

        # Use Vertical Lines
        self.use_vline_label = tk.Label(self.misc_frame, text="Use Vertical Lines")
        self.use_vline_label.grid(row=2, column=3)

        self.use_vline_status = tk.StringVar(value="No")
        self.use_vline_box = tk.Checkbutton(self.misc_frame, variable=self.use_vline_status, onvalue="Yes", offvalue="No")
        self.use_vline_box.grid(row=3, column=3)   
               
        
        # Load Data Button
        self.loaddat_button = tk.Button(self.button_frame, text="Load Data", command=lambda: self.button_load())
        self.loaddat_button.grid(row=4, column=0, padx=5)
        
        # Enter Data Button
        self.pltnfit_button = tk.Button(self.button_frame, text="Fit and Plot", command=lambda: self.fit_and_plot())
        self.pltnfit_button.grid(row=4, column=2, padx=5)
        
        # Save Data Button
        self.savedat_button = tk.Button(self.button_frame, text="Save Data", command=lambda: self.button_data_save())
        self.savedat_button.grid(row=4, column=4, padx=5)           
        
        # Save Plot Button
        self.saveplot_button = tk.Button(self.button_frame, text="Save Plot", command=lambda: self.button_plot_save())
        self.saveplot_button.grid(row=4, column=6, padx=5)
        
        # Enter Data Button
        self.exit_button = tk.Button(self.button_frame, text="Exit", command=lambda: self.exit_window())
        self.exit_button.grid(row=4, column=8, padx=5)
    
        self.data_loaded = 0
        self.data_fit = 0
        self.data_plotted = 0
    
        self.info_box.insert(1.0, 'Hello, Please Load a File')
        
        self.load_box.insert(1.0, 'Loaded File:')
        self.load_box.insert('end', '\n\nNone, Load a File to Proceed')
 
  
    def button_load(self):
        self.load_box.delete(2.0, 'end')        
        
        tk_root = Tk()
        tk_root.withdraw()
        filenames = filedialog.askopenfilenames(parent=None, title='Select Data file...')    
        tk_root.destroy()        
        
        self.file_in = filenames[0]        
        file_name_ext = os.path.basename(self.file_in)
        
        if self.file_in.endswith('.csv'):
            self.file_name = file_name_ext.replace('.csv', '')
            self.td, self.wave, self.spec = rhlib.read_rawdat(self.file_in)
            
            self.data_loaded = 1
            
            self.load_box.insert('end', f'\n\n{self.file_name}')
            self.info_box.insert('end', '\n\nFile Loaded Ready to Fit')
            self.info_box.see('end')
            
        elif self.file_in.endswith('.ufs'):
            self.file_name = file_name_ext.replace('.ufs', '')
            self.td, self.wave, self.spec = rhlib.openUFSfile(self.file_in)
            
            self.data_loaded = 1
            
            self.load_box.insert('end', f'\n\n{self.file_name}')
            self.info_box.insert('end', '\n\nFile Loaded Ready to Fit')
            self.info_box.see('end')
            
        else:
            self.info_box.insert('end', '\n\nWrong file type must be .CSV or .UFS')
            self.info_box.see('end') 
  
            
    def fit_and_plot(self):
        num_exp = int(self.num_exp_combobox.get())
        use_inf = self.use_inf_status.get()
        
        tz_val = float(self.tz_entry.get())
        irf_val = float(self.irf_entry.get())
            
        exp1_val = float(self.exp1_entry.get())
        exp2_val = float(self.exp2_entry.get())
        exp3_val = float(self.exp3_entry.get())
        exp4_val = float(self.exp4_entry.get())
        exp5_val = float(self.exp5_entry.get())
        exp6_val = float(self.exp6_entry.get())
        
        tzlb_val = float(self.tzlb_entry.get())
        irflb_val = float(self.irflb_entry.get())
        
        if self.fix_exp1_status.get() in ['Yes']:
            if float(self.exp1_entry.get()) == 0:
                exp1lb_val = float(self.exp1_entry.get())
                exp1ub_val = float(self.exp1_entry.get()) + 0.001
            elif float(self.exp1_entry.get()) > 0:
                exp1lb_val = float(self.exp1_entry.get()) - 0.001
                exp1ub_val = float(self.exp1_entry.get()) 
        else:      
             exp1lb_val = float(self.exp1lb_entry.get())
             if self.exp1ub_entry.get() in ['Inf','iNf','inF','INf','iNF', 'InF','INF','inf']:
                 exp1ub_val = np.inf
             else:
                 exp1ub_val = float(self.exp1ub_entry.get())
             
        if self.fix_exp2_status.get() in ['Yes']:
            if float(self.exp2_entry.get()) == 0:
                exp2lb_val = float(self.exp2_entry.get())
                exp2ub_val = float(self.exp2_entry.get()) + 0.001
            elif float(self.exp2_entry.get()) > 0:
                exp2lb_val = float(self.exp2_entry.get()) - 0.001
                exp2ub_val = float(self.exp2_entry.get()) 
        else:
            exp2lb_val = float(self.exp2lb_entry.get())
            if self.exp2ub_entry.get() in ['Inf','iNf','inF','INf','iNF', 'InF','INF','inf']:
                exp2ub_val = np.inf
            else:
                exp2ub_val = float(self.exp2ub_entry.get())
                
        if self.fix_exp3_status.get() in ['Yes']:
            if float(self.exp3_entry.get()) == 0:
                exp3lb_val = float(self.exp3_entry.get())
                exp3ub_val = float(self.exp3_entry.get()) + 0.001
            elif float(self.exp3_entry.get()) > 0:                
                exp3lb_val = float(self.exp3_entry.get()) - 0.001
                exp3ub_val = float(self.exp3_entry.get()) 
        else:
            exp3lb_val = float(self.exp3lb_entry.get())
            if self.exp3ub_entry.get() in ['Inf','iNf','inF','INf','iNF', 'InF','INF','inf']:
                exp3ub_val = np.inf
            else:
                exp3ub_val = float(self.exp3ub_entry.get())
                
        if self.fix_exp4_status.get() in ['Yes']:
            if float(self.exp4_entry.get()) == 0:
                exp4lb_val = float(self.exp4_entry.get())
                exp4ub_val = float(self.exp4_entry.get()) + 0.001
            elif float(self.exp4_entry.get()) > 0:
                exp4lb_val = float(self.exp4_entry.get()) - 0.001
                exp4ub_val = float(self.exp4_entry.get()) 
        else:
            exp4lb_val = float(self.exp4lb_entry.get())
            if self.exp4ub_entry.get() in ['Inf','iNf','inF','INf','iNF', 'InF','INF','inf']:
                exp4ub_val = np.inf
            else:
                exp4ub_val = float(self.exp4ub_entry.get())
                
        if self.fix_exp5_status.get() in ['Yes']:
            if float(self.exp5_entry.get()) == 0:
                exp5lb_val = float(self.exp5_entry.get())
                exp5ub_val = float(self.exp5_entry.get()) + 0.001
            elif float(self.exp5_entry.get()) > 0:
                exp5lb_val = float(self.exp5_entry.get()) - 0.001
                exp5ub_val = float(self.exp5_entry.get()) 
        else:
            exp5lb_val = float(self.exp5lb_entry.get())
            if self.exp5ub_entry.get() in ['Inf','iNf','inF','INf','iNF', 'InF','INF','inf']:
                exp5ub_val = np.inf
            else:
                exp5ub_val = float(self.exp5ub_entry.get())
                
        if self.fix_exp6_status.get() in ['Yes']:
            if float(self.exp6_entry.get()) == 0:
                exp6lb_val = float(self.exp6_entry.get())
                exp6ub_val = float(self.exp6_entry.get()) + 0.001
            elif float(self.exp6_entry.get()) > 0:
                exp6lb_val = float(self.exp6_entry.get()) - 0.001
                exp6ub_val = float(self.exp6_entry.get()) 
        else:
            exp6lb_val = float(self.exp6lb_entry.get())
            if self.exp6ub_entry.get() in ['Inf','iNf','inF','INf','iNF', 'InF','INF','inf']:
                exp6ub_val = np.inf 
            else:
                exp6ub_val = float(self.exp6ub_entry.get()) 
                
        tzub_val = float(self.tzub_entry.get())
        irfub_val = float(self.irfub_entry.get())
          
        
        if self.csfile_type_combobox.get() in ['Fill']:
            cpstyle = 0
        elif self.csfile_type_combobox.get() in ['Line']:
            cpstyle = 1
            
        if self.tunit_combobox.get() in ['fs']:
            tunit = 0
        elif self.tunit_combobox.get() in ['ps']:
            tunit = 1
        elif self.tunit_combobox.get() in ['ns']:
            tunit = 2
        elif self.tunit_combobox.get() in ['us']:
            tunit = 3
        
        if self.eunit_combobox.get() in ['nm']:
            eunit = 0
        elif self.eunit_combobox.get() in ['eV']:
            eunit = 1
           
        p_start = float(self.p_start_entry.get())
        s_start = float(self.s_start_entry.get())
        
        if self.dcmap_combobox.get() in ['Custom']:
            #   DADS Colormap               Gray        Red      Gold     Dk Blue     Pink     lt Blue     Purple
            dcmap = ListedColormap(['#939597','#ED1C24','#F7931D','#2E3092', '#ED2A7B', '#27A9E1', '#A128E0']) 
        elif self.dcmap_combobox.get() in ['Tab10']:
            dcmap = colormaps.get_cmap('tab10')
        
        use_vline = self.use_vline_status.get()
                    
        guess_set = [tz_val,irf_val,exp1_val,exp2_val,exp3_val,exp4_val,exp5_val,exp6_val]
        lowerb_set = [tzlb_val, irflb_val, exp1lb_val,exp2lb_val,exp3lb_val,exp4lb_val,exp5lb_val,exp6lb_val]
        upperb_set = [tzub_val, irfub_val, exp1ub_val,exp2ub_val,exp3ub_val,exp4ub_val,exp5ub_val,exp6ub_val]                   
        
        if self.data_loaded == 1:
            
            # Data Spacing
            dat_gap, gap_b, gap_val = rhlib.finddatagap(self.wave, len(self.wave))
            
            self.info_box.insert('end', '\n\nRunning Fit')
            
            # Run Fitting with error catching
            try:
                self.fit, fit_par, dads_a, lts, rcv, resid, EADS = GLA_fit(self.td, self.spec, num_exp, use_inf, guess_set, lowerb_set, upperb_set)
                 
                self.fig.axes[8].clear()
                self.fig.axes[7].clear()
                self.fig.axes[6].clear()
               
                # Clear Plot
                for i in range(2):
                    for j in range(3):
                        self.ax[i,j].clear()
               
                self.data_fit = 1
                
            except np.linalg.LinAlgError as err:
                if 'SVD did not converge in Linear Least Squares' in str(err):                                       
                    # Error Message
                    self.info_box.insert('end', '\n\nSVD did not converge, Try different Guess')
                    self.info_box.see('end')
                else:
                    raise
                    
            inf_sum = dads_a.sum(axis=0)

            #set up min time delay value, and time delay range/units
            if tunit == 1 or tunit == 3:
                td_plot = self.td
                lts_plot = lts
                
                tdmin = rhlib.timestart(self.td, p_start, 1.0)
                tdmin_s = rhlib.timestart(self.td, s_start, 1.0)
                
            else:
                td_plot = self.td*1000
                lts_plot = lts*1000
                
                tdmin = rhlib.timestart(self.td, p_start, 1000.0)
                tdmin_s = rhlib.timestart(self.td, s_start, 1000.0)


            
                    
            self.eads_fs = EADS.T
            
            # Setup Plots
            fitT = self.fit.T
            specT = self.spec.T
            residT = resid.T

            if dat_gap == 1:
            #Raw Data Setup Array with Gap for plotting    
                specTg = np.copy(specT)              
                specTg = np.insert(specTg, [gap_b+1], 'Nan', axis=1)
                fitTg = np.copy(fitT)              
                fitTg = np.insert(fitTg, [gap_b+1], 'Nan', axis=1)
                residTg = np.copy(residT)              
                residTg = np.insert(residTg, [gap_b+1], 'Nan', axis=1)
                waveg = np.insert(self.wave, [gap_b+1], gap_val, axis=0)
                waveg_plot = waveg
                wave_plot = self.wave
            else:
                waveg_plot = self.wave
                wave_plot = self.wave

            if eunit == 1:
                wave_plot = 1239.8/wave_plot
                
            
            # Parameters for Saving Data/Plot
            if use_inf in ['Yes']:
                self.t_dads = num_exp + 1
                self.dads_fs = np.copy(dads_a)
                self.dads_fs = np.insert(self.dads_fs, 0, 0, axis=1)
                self.dads_fs = np.insert(self.dads_fs, 0, 0, axis=1)
                self.save_par = np.copy(fit_par)
                self.save_par = np.insert(self.save_par, 2, 0, axis=0)
                self.eads_fs = np.insert(self.eads_fs, 0, 0, axis=1)
                self.eads_fs = np.insert(self.eads_fs, 0, 0, axis=1)
                self.eads_fs = np.insert(self.eads_fs, 0, 0, axis=1)
            else:
                self.t_dads = num_exp
                self.dads_fs = np.copy(dads_a)
                self.dads_fs = np.insert(self.dads_fs, 0, 0, axis=1)
                self.dads_fs = np.insert(self.dads_fs, 0, 0, axis=1)
                self.save_par = np.copy(fit_par)
                self.save_par = np.insert(self.save_par, 2, 0, axis=0)
                self.eads_fs = np.insert(self.eads_fs, 0, 0, axis=1)
                self.eads_fs = np.insert(self.eads_fs, 0, 0, axis=1)
                self.eads_fs = np.insert(self.eads_fs, 0, 0, axis=1)
                
            self.srcv = str(rcv)
            
            self.info_box.insert('end', '\n\nFitting Complete')
            
            self.info_box.insert('end', f'\n\nFit RCV: {self.srcv}')
            
            # Plot Parameters                
            time_scale  = ['fs', 'ps', 'ns', 'us']        
            c_map = ['gist_rainbow_r','RdBu_r', 'rainbow']        
            energy_scale = [r'Wavelength / nm', r'Photon Energy / eV']
            energy_units = ['nm', 'eV']
            
            self.info_box.insert('end', '\n\nPlotting Fit')
            
            # Plot Output           
            self.fig.suptitle(f'{self.file_name} - Fit Report')
            
            if cpstyle == 0:
            # Contour plots that are filled in    
                
                # Raw Data as contour plot
                if dat_gap == 1:
                    plt1 = rhlib.plot_TAScontfHeat_alt(specT, specTg, td_plot, waveg_plot, tdmin, r'Raw Data Surface', self.ax, 0,0, time_scale[tunit], c_map[0], energy_scale[eunit])
                else:
                    plt1 = rhlib.plot_TAScontfHeat_alt(specT, specT, td_plot, wave_plot, tdmin, r'Raw Data Surface', self.ax, 0,0, time_scale[tunit], c_map[0], energy_scale[eunit])
                            
                # Fit as contour plot
                if dat_gap == 1:
                    plt2 = rhlib.plot_TAScontfHeat_alt(fitT, fitTg, td_plot, waveg_plot, tdmin, r'Fit Surface', self.ax, 0,1, time_scale[tunit], c_map[0], energy_scale[eunit])
                else:
                    plt2 = rhlib.plot_TAScontfHeat_alt(fitT, fitT, td_plot, wave_plot, tdmin, r'Fit Surface', self.ax, 0,1, time_scale[tunit], c_map[0], energy_scale[eunit])
                            
                # Resid Surface as contour plot
                if dat_gap == 1:
                    plt3 = rhlib.plot_TAScontfHeat_alt(residT, residTg, td_plot, waveg_plot, tdmin, f'Residual Plot, RCV: {self.srcv}', self.ax, 0,2, time_scale[tunit], c_map[1], energy_scale[eunit])
                else:
                    plt3 = rhlib.plot_TAScontfHeat_alt(residT, residT, td_plot, wave_plot, tdmin, f'Residual Plot, RCV: {self.srcv}', self.ax, 0,2, time_scale[tunit], c_map[1], energy_scale[eunit])
            
            elif cpstyle == 1:
            # Contour plots that are just lines    
                
                if dat_gap == 1:
                    plt1 = rhlib.plot_TAScontHeat_alt(specT, specTg, td_plot, waveg_plot, tdmin, r'Raw Data Surface', self.ax, 0,0, time_scale[tunit], c_map[0], energy_scale[eunit])
                else:
                    plt1 = rhlib.plot_TAScontHeat_alt(specT, specT, td_plot, wave_plot, tdmin, r'Raw Data Surface', self.ax, 0,0, time_scale[tunit], c_map[0], energy_scale[eunit])
                            
                # Fit as contour plot
                if dat_gap == 1:
                    plt2 = rhlib.plot_TAScontHeat_alt(fitT, fitTg, td_plot, waveg_plot, tdmin, r'Fit Surface', self.ax, 0,1, time_scale[tunit], c_map[0], energy_scale[eunit])
                else:
                    plt2 = rhlib.plot_TAScontHeat_alt(fitT, fitT, td_plot, wave_plot, tdmin, r'Fit Surface', self.ax, 0,1, time_scale[tunit], c_map[0], energy_scale[eunit])
                            
                # Resid Surface as contour plot
                if dat_gap == 1:
                    plt3 = rhlib.plot_TAScontHeat_alt(residT, residTg, td_plot, waveg_plot, tdmin, f'Residual Plot, RCV: {self.srcv}', self.ax, 0,2, time_scale[tunit], c_map[1], energy_scale[eunit])
                else:
                    plt3 = rhlib.plot_TAScontHeat_alt(residT, residT, td_plot, wave_plot, tdmin, f'Residual Plot, RCV: {self.srcv}', self.ax, 0,2, time_scale[tunit], c_map[1], energy_scale[eunit])
                       
            self.cbar1 = self.fig.colorbar(plt1, cax=self.fig.axes[6])
            self.cbar2 = self.fig.colorbar(plt2, cax=self.fig.axes[7])
            self.cbar3 = self.fig.colorbar(plt3, cax=self.fig.axes[8])
            
            # Fit DADS Plot
            rhlib.plot_TASfitDADS_alt2(dads_a, wave_plot, lts_plot, inf_sum, dat_gap, gap_b, self.ax, 1,0, dcmap, time_scale[tunit], energy_scale[eunit])

            # Fit Representative Spectra
            rhlib.plot_TASRepSpectra_alt(self.spec, wave_plot, td_plot, tdmin_s, dat_gap, gap_b, self.ax, 1,1, 10, r'Raw Data Rep. Spectra', c_map[2], time_scale[tunit], energy_scale[eunit])

            # Single Kinetic Fits
            rhlib.plot_TASRandKinTrace_alt(specT, fitT, wave_plot, td_plot, tdmin, self.ax, 1,2, 'tab10', 6, time_scale[tunit], energy_units[eunit], use_vline, 0,0)
            
            self.canvas_plot.draw()
            self.data_plotted = 1
            
            self.info_box.insert('end', '\n\nPlotting Complete')
            self.info_box.see('end')
        else:
            self.info_box.insert('end', '\n\nNo Data Loaded, Please Load Data')
            self.info_box.see('end')
  
            
    def button_data_save(self):
        if self.data_fit == 1:
            rcv_print = self.srcv.replace('0.','')
            
            tk_root = Tk()
            tk_root.withdraw()
            self.write_dir = filedialog.askdirectory(title='Select folder to Save Data to')
            os.chdir(self.write_dir)
            tk_root.destroy()
            
            os.chdir(self.write_dir)
            rhlib.write_matrixdat(self.wave, self.td, self.fit, f'{self.file_name}_{str(self.t_dads)}_DADS_Fit_{rcv_print}.csv')
            rhlib.write_matrixdat(self.wave, self.save_par, self.dads_fs, f'{self.file_name}_DADS_{str(self.t_dads)}_{rcv_print}.csv')
            rhlib.write_matrixdat(self.wave, self.save_par, self.eads_fs, f'{self.file_name}_EADS_{str(self.t_dads)}_{rcv_print}.csv')
            
            self.info_box.insert('end', '\n\nData Saved')
            self.info_box.see('end')
        else:
            self.info_box.insert('end', '\n\nNo Data to Save, Please Run Fit')
            self.info_box.see('end')
    
        
    def button_plot_save(self):
        if self.data_plotted == 1:
            if self.pfile_type_combobox.get() in ['.PNG']:
                ext = '.png'
            elif self.pfile_type_combobox.get() in ['.SVG']:
                ext = '.svg'
                
            rcv_print = self.srcv.replace('0.','')
                    
            tk_root = Tk()
            tk_root.withdraw()
            self.write_dir = filedialog.askdirectory(title='Select folder to Save Plot to')
            os.chdir(self.write_dir)
            tk_root.destroy()
            
            os.chdir(self.write_dir)      
            self.fig.savefig(f'{self.file_name}_{str(self.t_dads)}_DADS_Fit_{rcv_print}{ext}')
              
            self.info_box.insert('end', f'\n\nPlot Saved as {ext} file')
            self.info_box.see('end')
        else:
            self.info_box.insert('end', '\n\nNo Plot to Save, Please Run Fit')
            self.info_box.see('end') 
    
    
    def exit_window(self):
        root.destroy()

root=tk.Tk()
app=Application()
app.mainloop()

