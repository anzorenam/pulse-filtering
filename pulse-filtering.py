#!/usr/bin/env python3.8
# -*- coding: utf8 -*-

import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import os

def treshold(PMT):
  THrs=np.array([[7.60e-3,14.40e-3,21.60e-3,28.80e-3],[8.80e-3,17.60e-3,25.60e-3,34.40e-3],[8.00e-3,15.60e-3,21.60e-3,29.60e-3],[8.00e-3,14.80e-3,20.80e-3,28.00e-3]])
  if PMT==1:
    s=THrs[0,:]
  elif PMT==2:
    s=THrs[1,:]
  elif PMT==3:
    s=THrs[2,:]
  elif PMT==4:
    s=THrs[3,:]
  return s

#this function makes constant fraction timing of a set of waveforms.

def cftiming(d,N,M,Fs,cf,t0,tf):
  s0=np.int(np.round(t0*Fs))
  sf=np.int(np.round(tf*Fs))
  st=s0+sf
  vmax=1.0/np.amax(d,axis=1)
  dnorm=d*np.transpose(vmax[np.newaxis])
  a=np.array([np.flatnonzero(dnorm[j,:]>=cf)[0]-1 for j in range(0,M)],dtype=np.int32)
  b=np.flatnonzero(np.logical_and(a+sf <= N,a-s0 >= 0))
  rt=np.size(b)
  dcf=np.zeros([rt,st],dtype=np.float)
  for j in range(0,rt):
    dcf[j,:]=d[b[j],a[b[j]]-s0:a[b[j]]+sf]
  return dcf,st,rt

mat.rc('text',usetex=True)
mat.rc('font',family="serif",serif="palatino")
mat.rcParams['text.latex.preamble']=[r'\usepackage[utf8]{inputenc}',r'\usepackage{mathpazo}',r'\usepackage[euler-digits,euler-hat-accent]{eulervm}',r'\usepackage[T1]{fontenc}',r'\usepackage[spanish]{babel}',r'\usepackage{amsmath,amsfonts,amssymb}',r'\usepackage{siunitx}'] 

plot=True # selecting if you want to read and process data or display results

# this parameters and functions could be ommited. I used to select a defined threshold for the signals
# depending on their source
PMT=1
ech=3
s=treshold(PMT)[ech-1]  

name='data.dat' # the data is assumed to be stored in text files.
                # each row of the file is a corresponding waveform

Fs=2.0e9 # Sampling frequency
x=np.genfromtxt(name,dtype=np.float,comments='FC:',delimiter=None,skip_header=2)

# Event selection
maxmax=np.amax(x)
x=x[np.all(x<maxmax,True)] # eliminate saturated waveforms
x=x[np.any(x>s,True)] # eliminate waverforms lower than the threshold
x=x[np.all(x[:,0:100]<5.0e-3,True)] # Optional. In this case samples [0:100] correspond to baseline.
                                    # Here we are elimininating all signals that doesn't have a constant baseline. 

#x=np.delete(x,122,0) used to eliminate indivual events 
N=np.size(x,1)
M=np.size(x,0)

# baseline fiting and removal
f0=np.mean(x[:,0:100])
x=x-np.transpose(f0[np.newaxis])

# filter desing
# I will use a FIR filter to reduce noise and preserve shape of the pulses.
k=np.arange(0,N,dtype=np.float)
w=(k/N)*Fs
ripp=20*np.log10(0.01)
bwidth=0.1
Ford,beta=signal.kaiserord(ripp,bwidth)
b=signal.firwin(Ford,0.1,window=('kaiser',beta))
y=signal.lfilter(b,1,x,axis=1)

# Constant fraction timing
t0,tf=10e-9,190e-9
ycf,Ncf,Mcf=cftiming(y,N,M,Fs,0.2,t0,tf)

# Plot results
if plot==True:
  wb,we=190,200 # selecting a few events to display. This is handy if the data set consists of a large number of waveforms.
  factor=1e9 # to normalize units
  ind=np.arange(N)
  tcf=factor*np.linspace(0,tf+t0,num=Ncf)
  fig=plt.figure(figsize=(8,4))
  ax=fig.add_subplot(1,1,1)
  ax.plot(tcf,np.transpose(ycf[wb:we+1,:]))
  plt.xlabel(r'$Time \left[\si{\nano\second}\right]$',x=0.95,horizontalalignment='right')
  plt.ylabel(r'$Amplitude \left[\si{\milli\volt}\right]$') # I use latex and package SIunits to make nice labels.
  
