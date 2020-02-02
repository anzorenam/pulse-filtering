#!/usr/bin/env python2.7
# -*- coding: utf8 -*-

import matplotlib as mat
import matplotlib.ticker as tick
import matplotlib.pyplot as plt
import numpy as np
import husl
import scipy.signal as signal
import os

def husl_gen():
  hue = np.random.randint(0, 360)
  saturation, lightness = np.random.randint(0, 100, 2)
  husl_dark = husl.husl_to_hex(hue, saturation, lightness/3)
  husl_light = husl.husl_to_hex(hue, saturation, lightness)
  return husl_dark, husl_light

def rplot(ax,x_range,data,**kwargs):
  husl_dark_hex, husl_light_hex = husl_gen()
  defaults = {'color':husl_dark_hex,'linewidth':1.0,'alpha':0.9}
  for x,y in defaults.iteritems():
    kwargs.setdefault(x, y)
  return ax.plot(x_range,data,**kwargs)

def rstyle(ax):
  ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
  ax.grid(True, 'minor', color='0.99', linestyle='-', linewidth=0.7)
  ax.patch.set_facecolor('0.90')
  ax.set_axisbelow(True)
  ax.xaxis.set_minor_locator((tick.MultipleLocator((plt.xticks()[0][1]-plt.xticks()[0][0]) / 2.0 )))
  ax.yaxis.set_minor_locator((tick.MultipleLocator((plt.yticks()[0][1]-plt.yticks()[0][0]) / 2.0 )))

  for child in ax.get_children():
    if isinstance(child, mat.spines.Spine):
      child.set_alpha(0)

  for line in ax.get_xticklines() + ax.get_yticklines():
    line.set_markersize(5)
    line.set_color("gray")
    line.set_markeredgewidth(1.4)

  for line in (ax.xaxis.get_ticklines(minor=True)+ax.yaxis.get_ticklines(minor=True)):
    line.set_markersize(0)

    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

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

fonts=15
rw='r'
PMT=1
ech=3
s=treshold(PMT)[ech-1]  
home=os.environ['HOME']
name='{0}/PDect/PMT{1}/131209.nw{2}'.format(home,PMT,ech)
f=open(name,'r')
tiempo=f.readline()[:-1][3:]
coefs=np.array(f.readline()[:-1].split()[1:5],dtype=np.float)
f.close()
Fs=1.0/coefs[2]
x=np.genfromtxt(name,dtype=np.float,comments='FC:',delimiter=None,skip_header=2)
maxmax=np.amax(x)
x=x[np.all(x<maxmax,True)]
x=x[np.any(x>s,True)]
x=x[np.all(x[:,0:100]<5.0e-3,True)]
#x=np.delete(x,122,0)  
N=np.size(x,1)
M=np.size(x,0)
f0=np.mean(x[:,0:100])
x=x-np.transpose(f0[np.newaxis])
k=np.arange(0,N,dtype=np.float)
w=(k/N)*Fs
ripp=20*np.log10(0.01)
bwidth=0.1
Ford,beta=signal.kaiserord(ripp,bwidth)
b=signal.firwin(Ford,0.1,window=('kaiser',beta))
y=signal.lfilter(b,1,x,axis=1)
t0,tf=10e-9,190e-9
ycf,Ncf,Mcf=cftiming(y,N,M,Fs,0.2,t0,tf)
  
if rw=='r':
  wb,we=190,200
  factor=1e9
  ind=np.arange(N)
  tcf=factor*np.linspace(0,tf+t0,num=Ncf)
  fig=plt.figure(figsize=(8,4))
  ax=fig.add_subplot(1,1,1)
  rplot(ax,tcf,np.transpose(ycf[wb:we+1,:]))
  rstyle(ax)
  plt.xlabel(r'$Tiempo \left[\si{\nano\second}\right]$',x=0.95,horizontalalignment='right',fontsize=fonts)
  plt.ylabel(r'$Amplitud \left[\si{\milli\volt}\right]$',fontsize=fonts)
  plt.ylim(1.05*np.amin(ycf[wb:we+1,:]),1.05*np.amax(ycf[wb:we+1,:]))
  plt.xlim(0,200)
  ax.tick_params(axis='both', which='major',labelsize=fonts)
  plt.savefig('pulsos-sinc.pdf',bbox_inches='tight',pad_inches=0.2)
