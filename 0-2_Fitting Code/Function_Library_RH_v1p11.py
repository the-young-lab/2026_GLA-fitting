# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:59:22 2023

@author: Robert Hamburger
"""

import numpy as np
import math 
import matplotlib as mpl
import matplotlib.ticker as plt_tk
import struct
from matplotlib import cm
from scipy import special

### FUNCTION LIST ##############################################################################

##################################################################################################
# define raw data load file format
def read_rawdat(filename):
    rdata = np.loadtxt(filename, delimiter=',')
    td    = rdata[0,1:]
    rwave = rdata[1:,0]
    rspec = rdata[1:, 1:]
    
    return td, rwave, rspec

###########################################################################################################
# define DADS data loading standard format
def read_dadsdat(filename):
    ddata = np.loadtxt(filename, delimiter=',')
    t_zero = ddata[0,0]
    irf    = ddata[1,0]
    lts    = ddata[2,1:]
    dwave  = ddata[3:,0]
    dads_a = ddata[3:,1:]
    
    return  t_zero, irf, lts, dwave, dads_a

##############################################################################################
# define function to write generic matrix output
def write_matrixdat(Wavelength, TimeDelay, Data, OutputName):
    out = np.zeros([len(Wavelength)+1, len(TimeDelay)+1])
    out[1:, 1:] = Data
    out[1:,0]   = Wavelength
    out[0,1:]   = TimeDelay
 
    np.savetxt(OutputName, out, delimiter=',')

##########################################################################################################
# decay function infinite
def decay_func_inf(TD, T_Zero, IRF):
    return ((100/2*np.exp(0)*np.exp(0))*(1+special.erf((TD-T_Zero-0)/(math.sqrt(0.4624)*IRF))))/100

#########################################################################################################
# decay function with IRF
def decay_func_irf(TD, T_Zero, IRF, Lt):
    return (100/2*np.exp(-((TD-T_Zero)/Lt))*np.exp(0.5*(IRF**2/Lt**2))*(1+special.erf((TD-T_Zero-(IRF**2/Lt))/(math.sqrt(0.4624)*IRF))))/100

#########################################################################################################
# decay function with IRF
def decay_func_irf(TD, T_Zero, IRF, Lt):
    return (100/2*np.exp(-((TD-T_Zero)/Lt))*np.exp(0.5*(IRF**2/Lt**2))*(1+special.erf((TD-T_Zero-(IRF**2/Lt))/(math.sqrt(0.4624)*IRF))))/100


###########################################################################################################
# decay function with no IRF
def decay_func(TD, T_Zero, Lt):
    return (100*np.exp(-((TD-T_Zero)/Lt)))/100

#############################################################################################################
# Define a fit model with only finite components
def GLA_model(guess, time_delay, data):
    # calculate the time evolution matrix
    decay_matrix = np.zeros((len(time_delay), len(guess)-2))
    for i in range(len(guess)-2):
        decay_matrix[:,i] = np.exp(-((time_delay-guess[0])/guess[i+2]))*np.exp(0.5*(guess[1]**2/guess[i+2]**2))*(1+special.erf((time_delay-guess[0]-(guess[1]**2/guess[i+2]))/(math.sqrt(0.4624)*guess[1])))
            
    # do the left-hand matrix divison to get the amplitude matrix
    dads_matrix = np.linalg.lstsq(decay_matrix, data.T, rcond=None)[0]
    
    # calculate the fits and residuals
    fit_matrix   = np.matmul(decay_matrix, dads_matrix).T
    
    return fit_matrix, dads_matrix, decay_matrix

#############################################################################################################
# Define a fit model with an infinite component
def GLA_model_i(guess, time_delay, data):
    # calculate the time evolution matrix
    decay_matrix = np.zeros((len(time_delay), len(guess)-1))
    for i in range(len(guess)-1):
        if i == 0:
            decay_matrix[:,i] = (np.exp(0)*np.exp(0))*(1+special.erf((time_delay-guess[0]-0)/(math.sqrt(0.4624)*guess[1])))
        else:
            decay_matrix[:,i] = np.exp(-((time_delay-guess[0])/guess[i+1]))*np.exp(0.5*(guess[1]**2/guess[i+1]**2))*(1+special.erf((time_delay-guess[0]-(guess[1]**2/guess[i+1]))/(math.sqrt(0.4624)*guess[1])))
            
    # do the left-hand matrix divison to get the amplitude matrix
    dads_matrix = np.linalg.lstsq(decay_matrix, data.T, rcond=None)[0]
    
    # calculate the fits and residuals
    fit_matrix   = np.matmul(decay_matrix, dads_matrix).T
    
    return fit_matrix, dads_matrix, decay_matrix

#############################################################################################################
# define a residual generator to send to the fitter
def GLA_residuals(guess, f, time_delay, data_matrix):
    fit_matrix = f(guess, time_delay, data_matrix)[0]
    resid_matrix = data_matrix - fit_matrix
    return np.reshape(resid_matrix, np.size(resid_matrix))

#############################################################################################################
# define function to find gaps in TAS matrix data, returns if gap exists, location of gap, and number at gap
def finddatagap(Data, n_Data):
    space = []
    for i in range(len(Data)):
        if i < n_Data-1:
            spacing = np.abs(Data[i] - Data[i+1])
            space.append(spacing)
        else:
            break

    spacing_max = np.amax(space)
    spacing_std = np.std(space)
    spacing_avg = sum(space)/len(space)

    if (spacing_max - spacing_avg) > spacing_std*2:
        dat_gap = 1
    else:
        dat_gap = 0

    for i in range(len(Data)):
        gap = np.abs(Data[i]-Data[i+1])
        if gap >= spacing_max:
            gap_b = i
            gap_val = (Data[i+1]-Data[i])/2 + Data[i] 
            break
        
    return dat_gap, gap_b, gap_val

################################################################################################
# define function to determine where the beginning time value should be for plotting
def timestart(TimeData, TimeStart, UnitMultiplier):
    td_plot = TimeData*UnitMultiplier
    for i in td_plot:
        if i > TimeStart:
            tdmin = i
            break
    return tdmin

################################################################################################
# define function to calculate RSE for a residual of two matrices
def matrixRSE(Data):
    rs =[]
    for row in range(len(Data)):
        for col in range(len(Data[0])):
            sqr = Data[row][col]**2
            rs.append(sqr)        
            
    rss = np.sum(rs)

    rse = math.sqrt(rss/(len(rs)-2))
    drse = round(rse,8)
    
    return drse

#############################################################################################
# define functions to read UFS file, and then open them and convert it to data matrix
def readString(filedata, cursor):
    string_length = struct.unpack_from('>I', filedata, offset=cursor)
    cursor += 4
    string = struct.unpack_from('>{}s'.format(string_length[0]), filedata, offset=cursor)
    cursor += string_length[0]
    return (string[0].decode('utf8'), cursor)

def readUInt(filedata, cursor):
    number = struct.unpack_from('>I', filedata, offset=cursor)
    cursor += 4
    return (number[0], cursor)

def readDoubles(filedata, cursor, count):
    numbers = struct.unpack_from('>{}d'.format(count), filedata, offset=cursor)
    cursor += 8*count
    return (list(numbers), cursor)

def openUFSfile(file_in):
    #extension = '.ufs'
    #filename = file_in + extension
    #filename = file_in
    f = open(file_in, 'rb')
    filedata = f.read()

    cursor = 0x0

    (version, cursor) = readString(filedata, cursor)

    (axis1_label, cursor) = readString(filedata, cursor)
    (axis1_units, cursor) = readString(filedata, cursor)
    (axis1_count, cursor) = readUInt(filedata, cursor)
    (axis1_data, cursor) = readDoubles(filedata, cursor, axis1_count)


    (axis2_label, cursor) = readString(filedata, cursor)
    (axis2_units, cursor) = readString(filedata, cursor)
    (axis2_count, cursor) = readUInt(filedata, cursor)
    (axis2_data, cursor) = readDoubles(filedata, cursor, axis2_count)


    (data_label, cursor) = readString(filedata, cursor)
    (data_size0, cursor) = readUInt(filedata, cursor)
    (data_size1, cursor) = readUInt(filedata, cursor)
    (data_size2, cursor) = readUInt(filedata, cursor)


    data_matrix = []
    for row in range(data_size1):
        (row_data, cursor) = readDoubles(filedata, cursor, data_size2)
        row_data.insert(0, axis1_data[row])
        #row_data = [x for x in row_data]
        data_matrix.append(row_data)

    axis2_data.insert(0, 0)
    data_matrix.insert(0, axis2_data)
    
    data_matrix2 = np.transpose(data_matrix)
    data_matrix3 = np.transpose(data_matrix2)
    
    data_matrix3[np.isnan(data_matrix3)] = 0
    
    td    = data_matrix3[0,1:]
    rwave = data_matrix3[1:,0]
    rspec = data_matrix3[1:, 1:]
    
    return td, rwave, rspec

#############################################################################################
# define plotting parameters for general "heat map" contour plots (Rainbow)
def plot_TAScontfHeat(Data, Data_Plot, TimeDelay, Wavelength, MinTime, Title, Axis, AxisNum, Time_Scale, CMap, Energy_Unit):
    plmax = np.amax(Data)
    plmin = np.amin(Data)
    if plmax > np.abs(plmin):
        vmin = -plmax
        vmax = plmax
    else:
        vmin = -np.abs(plmin)
        vmax = np.abs(plmin)
        
       
    Figure = Axis[AxisNum].contourf(Wavelength, TimeDelay, Data_Plot, 50, cmap=CMap, vmin=vmin, vmax=vmax)    
    Axis[AxisNum].set_ylabel('Time / ' + Time_Scale)
    Axis[AxisNum].set_xlabel(Energy_Unit)
    Axis[AxisNum].set_ylim([MinTime, TimeDelay[-1]])
    Axis[AxisNum].set_yscale('symlog')
    Axis[AxisNum].tick_params(direction='in', which='both')
    Axis[AxisNum].set_title(Title, fontsize=18)
    
    return Figure

#############################################################################################
# define plotting parameters for general "heat map" contour plots (Rainbow)
def plot_TAScontfHeat_alt(Data, Data_Plot, TimeDelay, Wavelength, MinTime, Title, Axis, AxisNumI, AxisNumJ, Time_Scale, CMap, Energy_Unit):
    plmax = np.amax(Data)
    plmin = np.amin(Data)
    if plmax > np.abs(plmin):
        vmin = -plmax
        vmax = plmax
    else:
        vmin = -np.abs(plmin)
        vmax = np.abs(plmin)
        
       
    Figure = Axis[AxisNumI,AxisNumJ].contourf(Wavelength, TimeDelay, Data_Plot, 50, cmap=CMap, vmin=vmin, vmax=vmax)    
    Axis[AxisNumI,AxisNumJ].set_ylabel('Time / ' + Time_Scale)
    Axis[AxisNumI,AxisNumJ].set_xlabel(Energy_Unit)
    Axis[AxisNumI,AxisNumJ].set_ylim([MinTime, TimeDelay[-1]])
    Axis[AxisNumI,AxisNumJ].set_yscale('symlog')
    Axis[AxisNumI,AxisNumJ].tick_params(direction='in', which='both')
    Axis[AxisNumI,AxisNumJ].set_title(Title, fontsize=18)
    
    return Figure

#############################################################################################
# define plotting parameters for general "heat map" contour plots (Rainbow)
def plot_TAScontHeat_alt(Data, Data_Plot, TimeDelay, Wavelength, MinTime, Title, Axis, AxisNumI, AxisNumJ, Time_Scale, CMap, Energy_Unit):
    plmax = np.amax(Data)
    plmin = np.amin(Data)
    if plmax > np.abs(plmin):
        vmin = -plmax
        vmax = plmax
    else:
        vmin = -np.abs(plmin)
        vmax = np.abs(plmin)
        
       
    Figure = Axis[AxisNumI,AxisNumJ].contour(Wavelength, TimeDelay, Data_Plot, 50, cmap=CMap, vmin=vmin, vmax=vmax)    
    Axis[AxisNumI,AxisNumJ].set_ylabel('Time / ' + Time_Scale)
    Axis[AxisNumI,AxisNumJ].set_xlabel(Energy_Unit)
    Axis[AxisNumI,AxisNumJ].set_ylim([MinTime, TimeDelay[-1]])
    Axis[AxisNumI,AxisNumJ].set_yscale('symlog')
    Axis[AxisNumI,AxisNumJ].tick_params(direction='in', which='both')
    Axis[AxisNumI,AxisNumJ].set_title(Title, fontsize=18)
    
    return Figure

#############################################################################################
# define plotting parameters for general "heat map" contour plots (Rainbow)
def plot_TAScontfHeat2(Data, Data_Plot, TimeDelay, Wavelength, MinTime, Title, Axis, AxisNum, Time_Scale, CMap, Energy_Unit):
    vmax = np.amax(Data)
    vmin = np.amin(Data)
           
    Figure = Axis[AxisNum].contourf(Wavelength, TimeDelay, Data_Plot, 50, cmap=CMap, vmin=vmin, vmax=vmax)    
    Axis[AxisNum].set_ylabel(Time_Scale)
    Axis[AxisNum].set_xlabel(Energy_Unit)
    Axis[AxisNum].set_ylim([MinTime, TimeDelay[-1]])
    Axis[AxisNum].set_yscale('symlog')
    Axis[AxisNum].set_xscale('symlog')
    Axis[AxisNum].xaxis.set_minor_locator(plt_tk.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12))
    Axis[AxisNum].yaxis.set_minor_locator(plt_tk.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12))
    Axis[AxisNum].tick_params(direction='in', which='both')
    Axis[AxisNum].set_title(Title, fontsize=18)
    
    return Figure

#############################################################################################
# define plotting parameters for general Residual contour plots (blue and red)
def plot_TAScontfResid(Data, TimeDelay, Wavelength, MinTime, Title, Axis, AxisNum, Time_Scale):
    plmax = np.amax(Data)
    plmin = np.amin(Data)
    if plmax > np.abs(plmin):
        vmin = -plmax
        vmax = plmax
    else:
        vmin = -np.abs(plmin)
        vmax = np.abs(plmin)
        
       
    Figure = Axis[AxisNum].contourf(Wavelength, TimeDelay, Data, 50, cmap='RdBu_r', vmin=vmin, vmax=vmax)    
    Axis[AxisNum].set_ylabel('Time / ' + Time_Scale)
    Axis[AxisNum].set_xlabel(r'Wavelength / nm')
    Axis[AxisNum].set_ylim([MinTime, TimeDelay[-1]])
    Axis[AxisNum].set_yscale('symlog')
    Axis[AxisNum].tick_params(direction='in', which='both')
    Axis[AxisNum].set_title(Title, fontsize=18)

    return Figure  

#############################################################################################
# define plotting parameters for plotting fit DADS for a given model
def plot_TASfitDADS(Data, Wavelength, Lifetimes, InfiniteSum, Gap, Gap_B, Axis, AxisNum, CMap, Time_Scale, Energy_Unit):
    if InfiniteSum[0] == 0 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            else:
                Axis[AxisNum].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNum].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    if InfiniteSum[0] == 0 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            else:
                Axis[AxisNum].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)            
    elif InfiniteSum[0] != 0 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNum].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label='inf')
                Axis[AxisNum].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
            else:
                Axis[AxisNum].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNum].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    elif InfiniteSum[0] != 0 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNum].plot(Wavelength, Data[:,i], color=CMap(i), label='inf')            
            else:
                Axis[AxisNum].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)            
    Axis[AxisNum].axhline(y=0.0, color='k', ls='-', lw=0.5)
    Axis[AxisNum].set_xlabel(Energy_Unit)
    Axis[AxisNum].set_ylabel(r'$\Delta$A / OD')
    if np.abs(Data.min()) < np.abs(Data.max()):
        Axis[AxisNum].set_ylim([-Data.max()*1.1, Data.max()*1.1])
    else:
        Axis[AxisNum].set_ylim([Data.min()*1.1, -Data.min()*1.1])
    if Wavelength[0] < Wavelength[-1]:
        Axis[AxisNum].set_xlim([Wavelength[0], Wavelength[-1]])
    else:
        Axis[AxisNum].set_xlim([Wavelength[-1], Wavelength[0]])
    Axis[AxisNum].legend(fontsize=8, ncol=2, loc='lower right')
    Axis[AxisNum].tick_params(direction='in', which='both')
    Axis[AxisNum].set_title(r'Fit DADS', fontsize=18)
#############################################################################################
# define plotting parameters for plotting fit DADS for a given model
def plot_TASfitDADS_alt(Data, Wavelength, Lifetimes, InfiniteSum, Gap, Gap_B, Axis, AxisNumI, AxisNumJ, CMap, Time_Scale, Energy_Unit):
    if InfiniteSum[0] == 0 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    if InfiniteSum[0] == 0 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)            
    elif InfiniteSum[0] != 0 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label='inf')
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    elif InfiniteSum[0] != 0 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(i), label='inf')            
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)            
    Axis[AxisNumI,AxisNumJ].axhline(y=0.0, color='k', ls='-', lw=0.5)
    Axis[AxisNumI,AxisNumJ].set_xlabel(Energy_Unit)
    Axis[AxisNumI,AxisNumJ].set_ylabel(r'$\Delta$A / OD')
    if np.abs(Data.min()) < np.abs(Data.max()):
        Axis[AxisNumI,AxisNumJ].set_ylim([-Data.max()*1.1, Data.max()*1.1])
    else:
        Axis[AxisNumI,AxisNumJ].set_ylim([Data.min()*1.1, -Data.min()*1.1])
    if Wavelength[0] < Wavelength[-1]:
        Axis[AxisNumI,AxisNumJ].set_xlim([Wavelength[0], Wavelength[-1]])
    else:
        Axis[AxisNumI,AxisNumJ].set_xlim([Wavelength[-1], Wavelength[0]])
    Axis[AxisNumI,AxisNumJ].legend(fontsize=8, ncol=2, loc='lower right')
    Axis[AxisNumI,AxisNumJ].tick_params(direction='in', which='both')
    Axis[AxisNumI,AxisNumJ].set_title(r'Fit DADS', fontsize=18)
        
#############################################################################################
# define plotting parameters for plotting fit DADS for a given model
def plot_TASfitDADS_alt2(Data, Wavelength, Lifetimes, InfiniteSum, Gap, Gap_B, Axis, AxisNumI, AxisNumJ, CMap, Time_Scale, Energy_Unit):
    
    # No Infinite, 1 Lifetime with Gap
    if InfiniteSum[0] == 0 and len(Lifetimes) == 2 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            elif i == 1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(1))
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    # No Infinite, 1< Lifetime with Gap           
    elif InfiniteSum[0] == 0 and len(Lifetimes) > 2 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            elif i == 1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(1))
            elif i == len(Lifetimes)-1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(0), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(0))
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    # Infinite, 1 Lifetime with Gap            
    elif InfiniteSum[0] != 0 and len(Lifetimes) == 2 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(0), label='unr')
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(0))
            elif i == 1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(1))
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    # Infinite, 1< Lifetime with Gap            
    elif InfiniteSum[0] != 0 and len(Lifetimes) > 2 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(0), label='unr')
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(0))
            elif i == 1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(1))
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    # No Infinite, 1 Lifetime, No Gap             
    elif InfiniteSum[0] == 0 and len(Lifetimes) == 2 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            elif i == 1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale) 
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
    # No Infinite, 1< Lifetime, No Gap
    elif InfiniteSum[0] == 0 and len(Lifetimes) > 2 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            elif i == 1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale) 
            elif i == len(Lifetimes)-1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(0), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
    # Infinite, 1 Lifetime, No Gap
    elif InfiniteSum[0] != 0 and len(Lifetimes) == 2 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(0), label='unr')
            elif i == 1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale) 
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
    # Infinite, 1< Lifetime, No Gap            
    elif InfiniteSum[0] != 0 and len(Lifetimes) > 2 and Gap != 1:
         for i in range(len(Lifetimes)): 
            if i == 0:
                 Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(0), label='unr')  
            elif i == 1:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale) 
            else:
                Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
  
    Axis[AxisNumI,AxisNumJ].axhline(y=0.0, color='k', ls='-', lw=0.5)
    Axis[AxisNumI,AxisNumJ].set_xlabel(Energy_Unit)
    Axis[AxisNumI,AxisNumJ].set_ylabel(r'$\Delta$A / OD')
    if np.abs(Data.min()) < np.abs(Data.max()):
        Axis[AxisNumI,AxisNumJ].set_ylim([-Data.max()*1.1, Data.max()*1.1])
    else:
        Axis[AxisNumI,AxisNumJ].set_ylim([Data.min()*1.1, -Data.min()*1.1])
    if Wavelength[0] < Wavelength[-1]:
        Axis[AxisNumI,AxisNumJ].set_xlim([Wavelength[0], Wavelength[-1]])
    else:
        Axis[AxisNumI,AxisNumJ].set_xlim([Wavelength[-1], Wavelength[0]])
    Axis[AxisNumI,AxisNumJ].legend(fontsize=8, ncol=2, loc='lower right')
    Axis[AxisNumI,AxisNumJ].tick_params(direction='in', which='both')
    Axis[AxisNumI,AxisNumJ].set_title(r'Fit DADS', fontsize=18)
    
#############################################################################################
# define plotting parameters for plotting fit DADS for a given model
def plot_TASfitDADSrc(Data, Wavelength, Lifetimes, InfiniteSum, Gap, Gap_B, Axis, AxisNum, CMap, Time_Scale, Energy_Unit):
    if InfiniteSum[0] == 0 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            else:
                Axis[AxisNum].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNum].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i-1))
    if InfiniteSum[0] == 0 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                pass
            else:
                Axis[AxisNum].plot(Wavelength, Data[:,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)            
    elif InfiniteSum[0] != 0 and Gap == 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNum].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label='inf')
                Axis[AxisNum].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
            else:
                Axis[AxisNum].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
                Axis[AxisNum].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
    elif InfiniteSum[0] != 0 and Gap != 1:
        for i in range(len(Lifetimes)):
            if i == 0:
                Axis[AxisNum].plot(Wavelength, Data[:,i], color=CMap(i), label='inf')            
            else:
                Axis[AxisNum].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)            
    Axis[AxisNum].axhline(y=0.0, color='k', ls='-', lw=0.5)
    Axis[AxisNum].set_xlabel(Energy_Unit)
    Axis[AxisNum].set_ylabel(r'$\Delta$A / OD')
    if np.abs(Data.min()) < np.abs(Data.max()):
        Axis[AxisNum].set_ylim([-Data.max()*1.1, Data.max()*1.1])
    else:
        Axis[AxisNum].set_ylim([Data.min()*1.1, -Data.min()*1.1])
    if Wavelength[0] < Wavelength[-1]:
        Axis[AxisNum].set_xlim([Wavelength[0], Wavelength[-1]])
    else:
        Axis[AxisNum].set_xlim([Wavelength[-1], Wavelength[0]])
    Axis[AxisNum].legend(fontsize=8, ncol=2, loc='lower right')
    Axis[AxisNum].tick_params(direction='in', which='both')
    Axis[AxisNum].set_title(r'Fit DADS', fontsize=18)
    
####################################################################################################################
# define plotting parameters for plotting represntative spectra generally
def plot_TASRepSpectra(Data, Wavelength, TimeDelay, MinTime, Gap, Gap_B, Axis, AxisNum, NumSpec, Title, ColorScale, Time_Scale, Energy_Unit):    
    sel_spec = np.logspace(np.log10(MinTime), np.log10(TimeDelay.max()), NumSpec) # selectedin ps
    ind_spec = np.zeros(len(sel_spec), dtype=int)
    spec_col = mpl.colormaps[ColorScale].resampled(len(sel_spec))
    if Gap == 1:
        for i in range(len(sel_spec)):
            ind_spec[i] = int(np.abs(TimeDelay-sel_spec[i]).argmin())
            Axis[AxisNum].plot(Wavelength[:Gap_B], Data[:Gap_B,ind_spec[i]], color=spec_col(i), label=r'%3.3f ' % TimeDelay[ind_spec[i]] + Time_Scale)
            Axis[AxisNum].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,ind_spec[i]], color=spec_col(i))
    else:
        for i in range(len(sel_spec)):
            ind_spec[i] = int(np.abs(TimeDelay-sel_spec[i]).argmin())
            Axis[AxisNum].plot(Wavelength, Data[:,ind_spec[i]], color=spec_col(i), label=r'%3.3f ' % TimeDelay[ind_spec[i]] + Time_Scale)       
    Axis[AxisNum].axhline(y=0.0, color='k', ls='-', lw=0.5)
    Axis[AxisNum].set_xlabel(Energy_Unit)
    Axis[AxisNum].set_ylabel(r'$\Delta$A / OD')
    if np.abs(Data.min()) < np.abs(Data.max()):
        Axis[AxisNum].set_ylim([-Data.max()*1.1, Data.max()*1.1])
    else:
        Axis[AxisNum].set_ylim([Data.min()*1.1, -Data.min()*1.1])
    if Wavelength[0] < Wavelength[-1]:
        Axis[AxisNum].set_xlim([Wavelength[0], Wavelength[-1]])
    else:
        Axis[AxisNum].set_xlim([Wavelength[-1], Wavelength[0]])
    Axis[AxisNum].legend(fontsize=8, ncol=2, loc='lower right')
    Axis[AxisNum].tick_params(direction='in', which='both')
    Axis[AxisNum].set_title(Title, fontsize=18)

####################################################################################################################
# define plotting parameters for plotting represntative spectra generally
def plot_TASRepSpectra_alt(Data, Wavelength, TimeDelay, MinTime, Gap, Gap_B, Axis, AxisNumI, AxisNumJ, NumSpec, Title, ColorScale, Time_Scale, Energy_Unit):    
    sel_spec = np.logspace(np.log10(MinTime), np.log10(TimeDelay.max()), NumSpec) # selectedin ps
    ind_spec = np.zeros(len(sel_spec), dtype=int)
    spec_col = mpl.colormaps[ColorScale].resampled(len(sel_spec))
    if Gap == 1:
        for i in range(len(sel_spec)):
            ind_spec[i] = int(np.abs(TimeDelay-sel_spec[i]).argmin())
            Axis[AxisNumI,AxisNumJ].plot(Wavelength[:Gap_B], Data[:Gap_B,ind_spec[i]], color=spec_col(i), label=r'%3.3f ' % TimeDelay[ind_spec[i]] + Time_Scale)
            Axis[AxisNumI,AxisNumJ].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,ind_spec[i]], color=spec_col(i))
    else:
        for i in range(len(sel_spec)):
            ind_spec[i] = int(np.abs(TimeDelay-sel_spec[i]).argmin())
            Axis[AxisNumI,AxisNumJ].plot(Wavelength, Data[:,ind_spec[i]], color=spec_col(i), label=r'%3.3f ' % TimeDelay[ind_spec[i]] + Time_Scale)       
    Axis[AxisNumI,AxisNumJ].axhline(y=0.0, color='k', ls='-', lw=0.5)
    Axis[AxisNumI,AxisNumJ].set_xlabel(Energy_Unit)
    Axis[AxisNumI,AxisNumJ].set_ylabel(r'$\Delta$A / OD')
    if np.abs(Data.min()) < np.abs(Data.max()):
        Axis[AxisNumI,AxisNumJ].set_ylim([-Data.max()*1.1, Data.max()*1.1])
    else:
        Axis[AxisNumI,AxisNumJ].set_ylim([Data.min()*1.1, -Data.min()*1.1])
    if Wavelength[0] < Wavelength[-1]:
        Axis[AxisNumI,AxisNumJ].set_xlim([Wavelength[0], Wavelength[-1]])
    else:
        Axis[AxisNumI,AxisNumJ].set_xlim([Wavelength[-1], Wavelength[0]])
    Axis[AxisNumI,AxisNumJ].legend(fontsize=8, ncol=2, loc='lower right')
    Axis[AxisNumI,AxisNumJ].tick_params(direction='in', which='both')
    Axis[AxisNumI,AxisNumJ].set_title(Title, fontsize=18)

############################################################################################################    
# define plotting parameters for plotting selected kinetic traces compared with fit model
def plot_TASRandKinTrace(RawData, FitData, Wavelength, TimeDelay, MinTime, Axis, AxisNum, CMap, NumSpec, Time_Scale, Energy_Unit, VLine, VLineAxis):
    dat_max = np.abs(RawData.max())
    dat_min = np.abs(RawData.min())
    fit_max = np.abs(FitData.max())
    fit_min = np.abs(FitData.min())
    sel_spec2 = np.logspace(np.log10(Wavelength.min()), np.log10(Wavelength.max()), NumSpec) # selectedin ps
    ind_spec2 = np.zeros(len(sel_spec2), dtype=int)
    #spec_col2 = cm.get_cmap(CMap, len(sel_spec2))
    spec_col2 = mpl.colormaps[CMap].resampled(len(sel_spec2))
    for i in range(len(sel_spec2)):
        ind_spec2[i] = int(np.abs(Wavelength-sel_spec2[i]).argmin())
        Axis[AxisNum].scatter(TimeDelay, RawData[:,ind_spec2[i]], s=10, color=spec_col2(i), label=r'%3.2f ' % Wavelength[ind_spec2[i]] + Energy_Unit)
        Axis[AxisNum].plot(TimeDelay, FitData[:,ind_spec2[i]], color=spec_col2(i), label=r'%3.2f ' % Wavelength[ind_spec2[i]] + Energy_Unit)
        if VLine in ['Yes','yEs','yeS','YEs','yES','YES','yes']: 
            Axis[VLineAxis].vlines(x=Wavelength[ind_spec2[i]], ymin=TimeDelay[0], ymax=TimeDelay[-1], color='k', ls='-', lw=0.5)            
    Axis[AxisNum].axhline(y=0.0, color='k', ls='-', lw=0.5)
    Axis[AxisNum].set_xlabel(r'Time / ' + Time_Scale)
    Axis[AxisNum].set_ylabel(r'$\Delta$A / OD')
    if dat_max >= dat_min and dat_max >= fit_max and dat_max >= fit_min:
        Axis[AxisNum].set_ylim([-dat_max*1.1, dat_max*1.1])
    elif dat_min >= dat_max and dat_min >= fit_max and dat_min >= fit_min:
        Axis[AxisNum].set_ylim([-dat_min*1.1, dat_min*1.1])
    elif fit_max >= dat_max and fit_max >= dat_min and fit_max >= fit_min:
        Axis[AxisNum].set_ylim([-fit_max*1.1, fit_max*1.1])
    else:
        Axis[AxisNum].set_ylim([-fit_min*1.1, fit_min*1.1])
    Axis[AxisNum].set_xlim([MinTime, TimeDelay[-1]])
    Axis[AxisNum].set_xscale('symlog')
    Axis[AxisNum].legend(fontsize=8, ncol=2, loc='lower right')
    Axis[AxisNum].tick_params(direction='in', which='both')
    Axis[AxisNum].set_title(r'Kinetic Traces', fontsize=18)

############################################################################################################    
# define plotting parameters for plotting selected kinetic traces compared with fit model
def plot_TASRandKinTrace_alt(RawData, FitData, Wavelength, TimeDelay, MinTime, Axis, AxisNumI, AxisNumJ, CMap, NumSpec, Time_Scale, Energy_Unit, VLine, VLineAxisI,VLineAxisJ):
    dat_max = np.abs(RawData.max())
    dat_min = np.abs(RawData.min())
    fit_max = np.abs(FitData.max())
    fit_min = np.abs(FitData.min())
    sel_spec2 = np.logspace(np.log10(Wavelength.min()), np.log10(Wavelength.max()), NumSpec) # selectedin ps
    ind_spec2 = np.zeros(len(sel_spec2), dtype=int)
    #spec_col2 = cm.get_cmap(CMap, len(sel_spec2))
    spec_col2 = mpl.colormaps[CMap].resampled(len(sel_spec2))
    for i in range(len(sel_spec2)):
        ind_spec2[i] = int(np.abs(Wavelength-sel_spec2[i]).argmin())
        Axis[AxisNumI,AxisNumJ].scatter(TimeDelay, RawData[:,ind_spec2[i]], s=10, color=spec_col2(i), label=r'%3.2f ' % Wavelength[ind_spec2[i]] + Energy_Unit)
        Axis[AxisNumI,AxisNumJ].plot(TimeDelay, FitData[:,ind_spec2[i]], color=spec_col2(i), label=r'%3.2f ' % Wavelength[ind_spec2[i]] + Energy_Unit)
        if VLine in ['Yes','yEs','yeS','YEs','yES','YES','yes']: 
            Axis[VLineAxisI,VLineAxisJ].vlines(x=Wavelength[ind_spec2[i]], ymin=TimeDelay[0], ymax=TimeDelay[-1], color='k', ls='-', lw=0.5)            
    Axis[AxisNumI,AxisNumJ].axhline(y=0.0, color='k', ls='-', lw=0.5)
    Axis[AxisNumI,AxisNumJ].set_xlabel(r'Time / ' + Time_Scale)
    Axis[AxisNumI,AxisNumJ].set_ylabel(r'$\Delta$A / OD')
    if dat_max >= dat_min and dat_max >= fit_max and dat_max >= fit_min:
        Axis[AxisNumI,AxisNumJ].set_ylim([-dat_max*1.1, dat_max*1.1])
    elif dat_min >= dat_max and dat_min >= fit_max and dat_min >= fit_min:
        Axis[AxisNumI,AxisNumJ].set_ylim([-dat_min*1.1, dat_min*1.1])
    elif fit_max >= dat_max and fit_max >= dat_min and fit_max >= fit_min:
        Axis[AxisNumI,AxisNumJ].set_ylim([-fit_max*1.1, fit_max*1.1])
    else:
        Axis[AxisNumI,AxisNumJ].set_ylim([-fit_min*1.1, fit_min*1.1])
    Axis[AxisNumI,AxisNumJ].set_xlim([MinTime, TimeDelay[-1]])
    Axis[AxisNumI,AxisNumJ].set_xscale('symlog')
    Axis[AxisNumI,AxisNumJ].legend(fontsize=8, ncol=2, loc='lower right')
    Axis[AxisNumI,AxisNumJ].tick_params(direction='in', which='both')
    Axis[AxisNumI,AxisNumJ].set_title(r'Kinetic Traces', fontsize=18)    
       

#######################################################################################################################
# define plotting parameters for plotting individual DADS in an exploded view
def plot_TASfitDADSex(Data, Wavelength, Lifetimes, nLifetimes, InfiniteSum, Gap, Gap_B, Axis, nMaxVal, nMinVal, CMap, Time_Scale, Energy_Unit):
    for i in range(len(Lifetimes)):
        if InfiniteSum[0] == 0 and i == 0: # skip placeholder if it exists
            pass
        elif InfiniteSum[0] != 0 and Gap == 1 and i == 0: # plot infinite with gap
            Axis[nLifetimes].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label='inf')
            Axis[nLifetimes].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
            
        elif InfiniteSum[0] != 0 and Gap != 1 and i == 0: # plot infinite without gap
            Axis[nLifetimes].plot(Wavelength, Data[:,i], color=CMap(i), label='inf')
            
        elif InfiniteSum[0] == 0 and Gap == 1 and i == 1: # plot long lifetime with gap
            Axis[nLifetimes-1].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            Axis[nLifetimes-1].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i-1))
            
        elif InfiniteSum[0] == 0 and Gap != 1 and i == 1: # plot long lifetime without gap
            Axis[nLifetimes-1].plot(Wavelength, Data[:,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            
        elif InfiniteSum[0] == 0 and Gap == 1 and i > 1: # plot dads with gap (no inf)
            Axis[i-1].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            Axis[i-1].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i-1))
           
        elif InfiniteSum[0] == 0 and Gap != 1 and i > 1: # plot dads without gap (no inf)
            Axis[i-1].plot(Wavelength, Data[:,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            
        elif InfiniteSum[0] != 0 and Gap == 1 and i > 0: # plot dads with gap (inf)
            Axis[i].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            Axis[i].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))            
        elif InfiniteSum[0] != 0 and Gap != 1 and i > 0: # plot dads without gap (inf)
            Axis[i].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)

    for i in range(len(Lifetimes)):
        if i == 0 and InfiniteSum[0] == 0: # Skip Placeholder
            pass
        elif i == 0 and InfiniteSum[0] != 0: # infinite DADS plot paramters
            Axis[nLifetimes].axhline(y=0.0, color='k', ls='-', lw=0.5)
            Axis[nLifetimes].set_xlabel(Energy_Unit)
            Axis[nLifetimes].set_ylabel(r'$\Delta$A / OD')
            if np.abs(nMinVal[i]) < np.abs(nMaxVal[i]):
                Axis[nLifetimes].set_ylim([-nMaxVal[i]*1.1, nMaxVal[i]*1.1])
            else:
                Axis[nLifetimes].set_ylim([nMinVal[i]*1.1, -nMinVal[i]*1.1])
            if Wavelength[0] < Wavelength[-1]:
                Axis[nLifetimes].set_xlim([Wavelength[0], Wavelength[-1]])
            else:
                Axis[nLifetimes].set_xlim([Wavelength[-1], Wavelength[0]])
            Axis[nLifetimes].legend(fontsize=8, ncol=2, loc='lower right')
            Axis[nLifetimes].tick_params(direction='in', which='both')
            Axis[nLifetimes].set_title(r'Infinite DADS', fontsize=18)    
        elif i == 1 and InfiniteSum[0] == 0: # first non infinite DADS
            life_num = str(nLifetimes)
            Axis[nLifetimes-1].axhline(y=0.0, color='k', ls='-', lw=0.5)
            Axis[nLifetimes-1].set_xlabel(Energy_Unit)
            Axis[nLifetimes-1].set_ylabel(r'$\Delta$A / OD')
            if np.abs(nMinVal[i]) < np.abs(nMaxVal[i]):
                Axis[nLifetimes-1].set_ylim([-nMaxVal[i]*1.1, nMaxVal[i]*1.1])
            else:
                Axis[nLifetimes-1].set_ylim([nMinVal[i]*1.1, -nMinVal[i]*1.1])
            if Wavelength[0] < Wavelength[-1]:
                Axis[nLifetimes-1].set_xlim([Wavelength[0], Wavelength[-1]])
            else:
                Axis[nLifetimes-1].set_xlim([Wavelength[-1], Wavelength[0]])
            Axis[nLifetimes-1].legend(fontsize=8, ncol=2, loc='lower right')
            Axis[nLifetimes-1].tick_params(direction='in', which='both')
            Axis[nLifetimes-1].set_title(r'Lifetime ' + life_num + ' DADS', fontsize=18)
        elif InfiniteSum[0] == 0: # no infinite present
            life_num = str(i-1) 
            Axis[i-1].axhline(y=0.0, color='k', ls='-', lw=0.5)
            Axis[i-1].set_xlabel(Energy_Unit)
            Axis[i-1].set_ylabel(r'$\Delta$A / OD')
            if np.abs(nMinVal[i]) < np.abs(nMaxVal[i]):
                Axis[i-1].set_ylim([-nMaxVal[i]*1.1, nMaxVal[i]*1.1])
            else:
                Axis[i-1].set_ylim([nMinVal[i]*1.1, -nMinVal[i]*1.1])
            if Wavelength[0] < Wavelength[-1]:
                Axis[i-1].set_xlim([Wavelength[0], Wavelength[-1]])
            else:
                Axis[i-1].set_xlim([Wavelength[-1], Wavelength[0]])
            Axis[i-1].legend(fontsize=8, ncol=2, loc='lower right')
            Axis[i-1].tick_params(direction='in', which='both')
            Axis[i-1].set_title(r'Lifetime ' + life_num + ' DADS', fontsize=18)
        else: # infinite present
            life_num = str(i) 
            Axis[i].axhline(y=0.0, color='k', ls='-', lw=0.5)
            Axis[i].set_xlabel(Energy_Unit)
            Axis[i].set_ylabel(r'$\Delta$A / OD')
            if np.abs(nMinVal[i]) < np.abs(nMaxVal[i]):
                Axis[i].set_ylim([-nMaxVal[i]*1.1, nMaxVal[i]*1.1])
            else:
                Axis[i].set_ylim([nMinVal[i]*1.1, -nMinVal[i]*1.1])
            if Wavelength[0] < Wavelength[-1]:
                Axis[i].set_xlim([Wavelength[0], Wavelength[-1]])
            else:
                Axis[i].set_xlim([Wavelength[-1], Wavelength[0]])
            Axis[i].legend(fontsize=8, ncol=2, loc='lower right')
            Axis[i].tick_params(direction='in', which='both')
            Axis[i].set_title(r'Lifetime ' + life_num + ' DADS', fontsize=18)

#######################################################################################################################
# define plotting parameters for plotting individual DADS in an exploded view
def plot_TASfitDADSnex(Data, Wavelength, Lifetimes, nLifetimes, InfiniteSum, Gap, Gap_B, Axis, CMap, Time_Scale, Energy_Unit):
    for i in range(len(Lifetimes)):
        if InfiniteSum[0] == 0 and i == 0: # skip placeholder if it exists
            pass
        elif InfiniteSum[0] != 0 and Gap == 1 and i == 0: # plot infinite with gap
            Axis[nLifetimes].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label='inf')
            Axis[nLifetimes].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))
            
        elif InfiniteSum[0] != 0 and Gap != 1 and i == 0: # plot infinite without gap
            Axis[nLifetimes].plot(Wavelength, Data[:,i], color=CMap(i), label='inf')
            
        elif InfiniteSum[0] == 0 and Gap == 1 and i == 1: # plot long lifetime with gap
            Axis[nLifetimes-1].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            Axis[nLifetimes-1].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i-1))
            
        elif InfiniteSum[0] == 0 and Gap != 1 and i == 1: # plot long lifetime without gap
            Axis[nLifetimes-1].plot(Wavelength, Data[:,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            
        elif InfiniteSum[0] == 0 and Gap == 1 and i > 1: # plot dads with gap (no inf)
            Axis[i-1].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            Axis[i-1].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i-1))
           
        elif InfiniteSum[0] == 0 and Gap != 1 and i > 1: # plot dads without gap (no inf)
            Axis[i-1].plot(Wavelength, Data[:,i], color=CMap(i-1), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            
        elif InfiniteSum[0] != 0 and Gap == 1 and i > 0: # plot dads with gap (inf)
            Axis[i].plot(Wavelength[:Gap_B], Data[:Gap_B,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)
            Axis[i].plot(Wavelength[Gap_B+1:], Data[Gap_B+1:,i], color=CMap(i))            
        elif InfiniteSum[0] != 0 and Gap != 1 and i > 0: # plot dads without gap (inf)
            Axis[i].plot(Wavelength, Data[:,i], color=CMap(i), label=r'%3.3f ' % Lifetimes[i] + Time_Scale)

    for i in range(len(Lifetimes)):
        if i == 0 and InfiniteSum[0] == 0: # Skip Placeholder
            pass
        elif i == 0 and InfiniteSum[0] != 0: # infinite DADS plot paramters
            Axis[nLifetimes].axhline(y=0.0, color='k', ls='-', lw=0.5)
            Axis[nLifetimes].set_xlabel(Energy_Unit)
            Axis[nLifetimes].set_ylabel(r'$\Delta$A / OD')
            if np.abs(Data.min()) < np.abs(Data.max()):
                Axis[nLifetimes].set_ylim([-Data.max()*1.1, Data.max()*1.1])
            else:
                Axis[nLifetimes].set_ylim([Data.min()*1.1, -Data.min()*1.1])
            if Wavelength[0] < Wavelength[-1]:
                Axis[nLifetimes].set_xlim([Wavelength[0], Wavelength[-1]])
            else:
                Axis[nLifetimes].set_xlim([Wavelength[-1], Wavelength[0]])
            Axis[nLifetimes].legend(fontsize=8, ncol=2, loc='lower right')
            Axis[nLifetimes].tick_params(direction='in', which='both')
            Axis[nLifetimes].set_title(r'Infinite DADS', fontsize=18)    
        elif i == 1 and InfiniteSum[0] == 0: # first non infinite DADS
            life_num = str(nLifetimes)
            Axis[nLifetimes-1].axhline(y=0.0, color='k', ls='-', lw=0.5)
            Axis[nLifetimes-1].set_xlabel(Energy_Unit)
            Axis[nLifetimes-1].set_ylabel(r'$\Delta$A / OD')
            if np.abs(Data.min()) < np.abs(Data.max()):
                Axis[nLifetimes-1].set_ylim([-Data.max()*1.1, Data.max()*1.1])
            else:
                Axis[nLifetimes-1].set_ylim([Data.min()*1.1, -Data.min()*1.1])
            if Wavelength[0] < Wavelength[-1]:
                Axis[nLifetimes-1].set_xlim([Wavelength[0], Wavelength[-1]])
            else:
                Axis[nLifetimes-1].set_xlim([Wavelength[-1], Wavelength[0]])
            Axis[nLifetimes-1].legend(fontsize=8, ncol=2, loc='lower right')
            Axis[nLifetimes-1].tick_params(direction='in', which='both')
            Axis[nLifetimes-1].set_title(r'Lifetime ' + life_num + ' DADS', fontsize=18)
        elif InfiniteSum[0] == 0: # no infinite present
            life_num = str(i-1) 
            Axis[i-1].axhline(y=0.0, color='k', ls='-', lw=0.5)
            Axis[i-1].set_xlabel(Energy_Unit)
            Axis[i-1].set_ylabel(r'$\Delta$A / OD')
            if np.abs(Data.min()) < np.abs(Data.max()):
                Axis[i-1].set_ylim([-Data.max()*1.1, Data.max()*1.1])
            else:
                Axis[i-1].set_ylim([Data.min()*1.1, -Data.min()*1.1])
            if Wavelength[0] < Wavelength[-1]:
                Axis[i-1].set_xlim([Wavelength[0], Wavelength[-1]])
            else:
                Axis[i-1].set_xlim([Wavelength[-1], Wavelength[0]])
            Axis[i-1].legend(fontsize=8, ncol=2, loc='lower right')
            Axis[i-1].tick_params(direction='in', which='both')
            Axis[i-1].set_title(r'Lifetime ' + life_num + ' DADS', fontsize=18)
        else: # infinite present
            life_num = str(i) 
            Axis[i].axhline(y=0.0, color='k', ls='-', lw=0.5)
            Axis[i].set_xlabel(Energy_Unit)
            Axis[i].set_ylabel(r'$\Delta$A / OD')
            if np.abs(Data.min()) < np.abs(Data.max()):
                Axis[i].set_ylim([-Data.max()*1.1, Data.max()*1.1])
            else:
                Axis[i].set_ylim([Data.min()*1.1, -Data.min()*1.1])
            if Wavelength[0] < Wavelength[-1]:
                Axis[i].set_xlim([Wavelength[0], Wavelength[-1]])
            else:
                Axis[i].set_xlim([Wavelength[-1], Wavelength[0]])
            Axis[i].legend(fontsize=8, ncol=2, loc='lower right')
            Axis[i].tick_params(direction='in', which='both')
            Axis[i].set_title(r'Lifetime ' + life_num + ' DADS', fontsize=18)