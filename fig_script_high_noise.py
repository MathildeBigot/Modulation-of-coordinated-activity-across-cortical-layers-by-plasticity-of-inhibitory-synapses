from matplotlib import rcParams

from numpy import *

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import cmath
from random import gauss
from scipy.optimize import curve_fit

##########################################

fig_width = 14 # width in inches

fig_height = 6  # height in inches

fig_size =  [fig_width,fig_height]

params = {'axes.labelsize': 16,

'axes.titlesize': 16,

'legend.fontsize': 11,

'font.size': 11,

'xtick.labelsize': 12,

'ytick.labelsize': 12,

'figure.figsize': fig_size,

'savefig.dpi' : 600,

'axes.linewidth' : 1.3,

'ytick.major.size' : 4,      # major tick size in points

'xtick.major.size' : 4      # major tick size in points
          
#'edgecolor' : None
          
#'xtick.major.size' : 2,
          
#'ytick.major.size' : 2,
          
}

rcParams.update(params)


# set sans-serif font to Arial

rcParams['font.sans-serif'] = 'Arial'



#########################Model
T       = 0.2                 # s
dt      = 0.0005              
time    = arange(0, T+dt, dt)

tau_e = 0.002                     # s
K0_e = 20                  
tau_i = 0.007                     # s
deltasyn = 0.002                  # s

gm      = 10E-9                   # conductance (S)
Cm      = 100E-12                # capacitance (F)
tau_m   = Cm/gm               # time constant (sec)

Vth     = 0.040                # spike threshold (V)
V_spike = 0.080                # spike delta (V)
V_reset = 0.000               # reset potential (V)                    

####Synaptic filter (B)

K1 = zeros(len(time))
K2 = zeros(len(time))
K3 = zeros(len(time))
K4 = zeros(len(time))

K_e = zeros(len(time)) 
K_i = zeros(len(time)) 

#examples for graph

K0_i = 1

for i, t in enumerate(time):
    K_e[i] = K0_e*exp(-t/tau_e)
    if t < deltasyn:
      K_i[i] = 0
    else:
      K_i[i] = K0_i*exp(-(t-deltasyn)/tau_i)
    K1[i]=K_e[i]-K_i[i]


K0_i = 8

for i, t in enumerate(time):
    K_e[i] = K0_e*exp(-t/tau_e)
    if t < deltasyn:
      K_i[i] = 0
    else:
      K_i[i] = K0_i*exp(-(t-deltasyn)/tau_i)
    K2[i]=K_e[i]-K_i[i]

K0_i = 16

for i, t in enumerate(time):
    K_e[i] = K0_e*exp(-t/tau_e)
    if t < deltasyn:
      K_i[i] = 0
    else:
      K_i[i] = K0_i*exp(-(t-deltasyn)/tau_i)
    K3[i]=K_e[i]-K_i[i]

K0_i = 24

for i, t in enumerate(time):
    K_e[i] = K0_e*exp(-t/tau_e)
    if t < deltasyn:
      K_i[i] = 0
    else:
      K_i[i] = K0_i*exp(-(t-deltasyn)/tau_i)
    K4[i]=K_e[i]-K_i[i]

############Activity examples (C)

a=4 #temps (nombre de dt) sur lequel on moyenne les spikes
time2=arange(0, T+dt, a*dt)

omega = 60*pi                  # rad/s
r_omega = 3E-9 #3E-9

iter=10000#3000  #nombre de trials pour moyenner les spikes

moyenne=zeros(len(time))

#####bruit synaptique
mu=-.61#.65#3.2
SD=2.0#1
noise = array([zeros(len(time))]*iter) 

#examples for graph
####################################################################################
#Ki/Ke = 0.05
j=complex(0,1)

I_post0 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm0      = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials0=array([zeros(len(time))]*iter)  #matrice compteur de spikes
rate0=zeros(len(time)//a+1)

Ki = 1
K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))
Ipost_wout_noise0=linspace(complex(0,0),complex(0,0),len(time))
I_pre=linspace(0,0,len(time))


for i in range(iter):
    Vm0[0:iter,-1]=-0.5
    for j, t in enumerate(time):
      noise[i,j]=gauss(mu,SD)
      I_post0[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
      Ipost_wout_noise0[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))
      I_pre[j] = r_omega*cos(omega*t)

# membrane potential variations 
      if Vm0[i,j-1] >= V_spike:
        Vm0[i,j] = V_reset
      else:
        Vm0[i,j] = (-Vm0[i,j-1] + I_post0[i,j-1]/gm)*dt/tau_m + Vm0[i,j-1]
      if Vm0[i,j] >= Vth:
        Vm0[i,j] = V_spike

#compteur spikes
      if Vm0[i,j] >= V_spike:
        mat_trials0[i,j] = 1
      else:
        mat_trials0[i,j]=0

raster0=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials0[i,j] == 1:
        raster0[0].append(j)
        raster0[1].append(i)

#

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials0[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate0[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 1.98
guess_phase = -0.35*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit0(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate0)

fit_0 = curve_fit(fit0, time2, rate0, p0=p0)
data_first_guess = fit0(time2, *p0)
data_fit0 = fit0(time2, *fit_0[0])

####################################

#Ki/Ke = 0.2
j=complex(0,1)

I_post_plus1 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm_plus1      = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials_plus1=array([zeros(len(time))]*iter)  #matrice compteur de spikes
rate_plus1=zeros(len(time)//a+1)

Ki = 4
K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))
Ipost_wout_noise_plus1=linspace(complex(0,0),complex(0,0),len(time))


for i in range(iter):
    Vm_plus1[0:iter,-1]=-0.5
    for j, t in enumerate(time):
      #noise[i,j]=gauss(mu,SD)
      I_post_plus1[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
      Ipost_wout_noise_plus1[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))

# membrane potential variations 
      if Vm_plus1[i,j-1] >= V_spike:
        Vm_plus1[i,j] = V_reset
      else:
        Vm_plus1[i,j] = (-Vm_plus1[i,j-1] + I_post_plus1[i,j-1]/gm)*dt/tau_m + Vm_plus1[i,j-1]
      if Vm_plus1[i,j] >= Vth:
        Vm_plus1[i,j] = V_spike

#compteur spikes
      if Vm_plus1[i,j] >= V_spike:
        mat_trials_plus1[i,j] = 1
      else:
        mat_trials_plus1[i,j]=0

raster_plus1=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials_plus1[i,j] == 1:
        raster_plus1[0].append(j)
        raster_plus1[1].append(i)

#

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials_plus1[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate_plus1[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 1.98
guess_phase = -0.35*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit_plus1(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate_plus1)

fit__plus1 = curve_fit(fit_plus1, time2, rate_plus1, p0=p0)
data_first_guess = fit_plus1(time2, *p0)
data_fit_plus1 = fit_plus1(time2, *fit__plus1[0])

###########################################################
#Ki/Ke = 0.4
j=complex(0,1)

I_post1 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm1      = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials1=array([zeros(len(time))]*iter)  #matrice compteur de spikes
rate1=zeros(len(time)//a+1)

Ki = 8
K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))
Ipost_wout_noise1=linspace(complex(0,0),complex(0,0),len(time))

for i in range(iter):
  Vm1[0:iter,-1]=-0.5
  for j, t in enumerate(time):
    #noise[i,j]=gauss(mu,SD)
    I_post1[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
    Ipost_wout_noise1[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))

# membrane potential variations 
    if Vm1[i,j-1] >= V_spike:
      Vm1[i,j] = V_reset
    else:
      Vm1[i,j] = (-Vm1[i,j-1] + I_post1[i,j-1]/gm)*dt/tau_m + Vm1[i,j-1]
    if Vm1[i,j] >= Vth:
      Vm1[i,j] = V_spike


#compteur spikes
    if Vm1[i,j] >= V_spike:
      mat_trials1[i,j] = 1
    else:
      mat_trials1[i,j]=0

raster1=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials1[i,j] == 1:
        raster1[0].append(j)
        raster1[1].append(i)

#

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials1[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate1[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 1.7
guess_phase = -0.01*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit1(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate1)

fit_1 = curve_fit(fit1, time2, rate1, p0=p0)
data_first_guess = fit1(time2, *p0)
data_fit1 = fit1(time2, *fit_1[0])

###########################################################
#Ki/Ke = 0.6
j=complex(0,1)

I_post_plus2 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm_plus2      = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials_plus2=array([zeros(len(time))]*iter)  #matrice compteur de spikes
rate_plus2=zeros(len(time)//a+1)

Ki = 12
K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))
Ipost_wout_noise_plus2=linspace(complex(0,0),complex(0,0),len(time))

for i in range(iter):
  Vm_plus2[0:iter,-1]=-0.5
  for j, t in enumerate(time):
    #noise[i,j]=gauss(mu,SD)
    I_post_plus2[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
    Ipost_wout_noise_plus2[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))

# membrane potential variations 
    if Vm_plus2[i,j-1] >= V_spike:
      Vm_plus2[i,j] = V_reset
    else:
      Vm_plus2[i,j] = (-Vm_plus2[i,j-1] + I_post_plus2[i,j-1]/gm)*dt/tau_m + Vm_plus2[i,j-1]
    if Vm_plus2[i,j] >= Vth:
      Vm_plus2[i,j] = V_spike


#compteur spikes
    if Vm_plus2[i,j] >= V_spike:
      mat_trials_plus2[i,j] = 1
    else:
      mat_trials_plus2[i,j]=0

raster_plus2=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials_plus2[i,j] == 1:
        raster_plus2[0].append(j)
        raster_plus2[1].append(i)

#

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials_plus2[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate_plus2[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 1.7
guess_phase = -0.01*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit_plus2(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate_plus2)

fit__plus2 = curve_fit(fit_plus2, time2, rate_plus2, p0=p0)
data_first_guess = fit1(time2, *p0)
data_fit_plus2 = fit_plus2(time2, *fit__plus2[0])

############################################################
#Ki/Ke=0.8
j=complex(0,1)

I_post2 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm2      = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials2=array([zeros(len(time))]*iter)
rate2=zeros(len(time)//a+1)
Ipost_wout_noise2=linspace(complex(0,0),complex(0,0),len(time))

Ki = 16

K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))

for i in range(iter):
  Vm2[0:iter,-1]=-0.5
  for j, t in enumerate(time):
    #noise[i,j]=gauss(mu,SD)                             
    I_post2[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
    Ipost_wout_noise2[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))

# membrane potential variations 
    if Vm2[i,j-1] >= V_spike:
      Vm2[i,j] = V_reset
    else:
      Vm2[i,j] = (-Vm2[i,j-1] + I_post2[i,j-1]/gm)*dt/tau_m + Vm2[i,j-1]
    if Vm2[i,j] >= Vth:
      Vm2[i,j] = V_spike
#compteur spikes
    if Vm2[i,j] >= V_spike:
      mat_trials2[i,j] = 1
    else:
      mat_trials2[i,j]=0

raster2=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials2[i,j] == 1:
        raster2[0].append(j)
        raster2[1].append(i)

#

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials2[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate2[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 2.39
guess_phase = 0.14*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit2(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate2)

fit_2 = curve_fit(fit2, time2, rate2, p0=p0)
data_first_guess = fit2(time2, *p0)
data_fit2 = fit2(time2, *fit_2[0])

############################################################
#Ki/Ke=1
j=complex(0,1)

I_post_plus3 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm_plus3      = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials_plus3=array([zeros(len(time))]*iter)
rate_plus3=zeros(len(time)//a+1)
Ipost_wout_noise_plus3=linspace(complex(0,0),complex(0,0),len(time))

Ki = 20

K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))

for i in range(iter):
  Vm_plus3[0:iter,-1]=-0.5
  for j, t in enumerate(time):
    #noise[i,j]=gauss(mu,SD)                          
    I_post_plus3[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
    Ipost_wout_noise_plus3[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))

# membrane potential variations 
    if Vm_plus3[i,j-1] >= V_spike:
      Vm_plus3[i,j] = V_reset
    else:
      Vm_plus3[i,j] = (-Vm_plus3[i,j-1] + I_post_plus3[i,j-1]/gm)*dt/tau_m + Vm_plus3[i,j-1]
    if Vm_plus3[i,j] >= Vth:
      Vm_plus3[i,j] = V_spike
#compteur spikes
    if Vm_plus3[i,j] >= V_spike:
      mat_trials_plus3[i,j] = 1
    else:
      mat_trials_plus3[i,j]=0

raster_plus3=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials_plus3[i,j] == 1:
        raster_plus3[0].append(j)
        raster_plus3[1].append(i)

#

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials_plus3[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate_plus3[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 2.39
guess_phase = 0.14*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit_plus3(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate_plus3)

fit__plus3 = curve_fit(fit_plus3, time2, rate_plus3, p0=p0)
data_first_guess = fit_plus3(time2, *p0)
data_fit_plus3 = fit_plus3(time2, *fit__plus3[0])

#############################################################
#Ki/Ke = 1.2
j=complex(0,1)

I_post3 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm3      = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials3=array([zeros(len(time))]*iter)
rate3=zeros(len(time)//a+1)
Ipost_wout_noise3=linspace(complex(0,0),complex(0,0),len(time))

Ki = 24

K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))

for i in range(iter):
  Vm3[0:iter,-1]=-0.5
  for j, t in enumerate(time):
    #noise[i,j]=gauss(mu,SD)                   
    I_post3[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
    Ipost_wout_noise3[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))

# membrane potential variations 
    if Vm3[i,j-1] >= V_spike:
      Vm3[i,j] = V_reset
    else:
      Vm3[i,j] = (-Vm3[i,j-1] + I_post3[i,j-1]/gm)*dt/tau_m + Vm3[i,j-1]
    if Vm3[i,j] >= Vth:
      Vm3[i,j] = V_spike
#compteur spikes
    if Vm3[i,j] >= V_spike:
      mat_trials3[i,j] = 1
    else:
      mat_trials3[i,j]=0

raster3=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials3[i,j] == 1:
        raster3[0].append(j)
        raster3[1].append(i)

#

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials3[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate3[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 4.01
guess_phase = 0.24*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit3(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate3)

fit_3 = curve_fit(fit3, time2, rate3, p0=p0)
data_first_guess = fit3(time2, *p0)
data_fit3 = fit3(time2, *fit_3[0])

#############################################################
#Ki/Ke = 1.4
j=complex(0,1)

I_post_plus4 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm_plus4      = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials_plus4=array([zeros(len(time))]*iter)
rate_plus4=zeros(len(time)//a+1)
Ipost_wout_noise_plus4=linspace(complex(0,0),complex(0,0),len(time))

Ki = 28

K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))

for i in range(iter):
  Vm_plus4[0:iter,-1]=-0.5
  for j, t in enumerate(time):
    #noise[i,j]=gauss(mu,SD)                   
    I_post_plus4[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
    Ipost_wout_noise_plus4[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))

# membrane potential variations 
    if Vm_plus4[i,j-1] >= V_spike:
      Vm_plus4[i,j] = V_reset
    else:
      Vm_plus4[i,j] = (-Vm_plus4[i,j-1] + I_post_plus4[i,j-1]/gm)*dt/tau_m + Vm_plus4[i,j-1]
    if Vm_plus4[i,j] >= Vth:
      Vm_plus4[i,j] = V_spike
#compteur spikes
    if Vm_plus4[i,j] >= V_spike:
      mat_trials_plus4[i,j] = 1
    else:
      mat_trials_plus4[i,j]=0

raster_plus4=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials_plus4[i,j] == 1:
        raster_plus4[0].append(j)
        raster_plus4[1].append(i)

#

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials_plus4[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate_plus4[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 4.01
guess_phase = 0.24*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit_plus4(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate_plus4)

fit__plus4 = curve_fit(fit_plus4, time2, rate_plus4, p0=p0)
data_first_guess = fit_plus4(time2, *p0)
data_fit_plus4 = fit_plus4(time2, *fit__plus4[0])


#############################################################
#Ki/Ke = 1.55
j=complex(0,1)

I_post_plus5 = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
Vm_plus5     = array([linspace(complex(0,0),complex(0,0),len(time))]*iter)
mat_trials_plus5=array([zeros(len(time))]*iter)
rate_plus5=zeros(len(time)//a+1)
Ipost_wout_noise_plus5=linspace(complex(0,0),complex(0,0),len(time))

Ki = 31

K=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))

for i in range(iter):
  Vm_plus5[0:iter,-1]=-0.5
  for j, t in enumerate(time):
    #noise[i,j]=gauss(mu,SD)                   
    I_post_plus5[i,j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K)) + SD*sqrt(Cm*gm)*noise[i,j]     #A
    Ipost_wout_noise_plus5[j]=sqrt(2*pi)*r_omega*abs(K)*cos(omega*t+cmath.phase(K))

# membrane potential variations 
    if Vm_plus5[i,j-1] >= V_spike:
      Vm_plus5[i,j] = V_reset
    else:
      Vm_plus5[i,j] = (-Vm_plus5[i,j-1] + I_post_plus5[i,j-1]/gm)*dt/tau_m + Vm_plus5[i,j-1]
    if Vm_plus5[i,j] >= Vth:
      Vm_plus5[i,j] = V_spike
#compteur spikes
    if Vm_plus5[i,j] >= V_spike:
      mat_trials_plus5[i,j] = 1
    else:
      mat_trials_plus5[i,j]=0

raster_plus5=[list(),list()]
   
for i in range(iter):
    for j, t in enumerate(time):
      if mat_trials_plus5[i,j] == 1:
        raster_plus5[0].append(j)
        raster_plus5[1].append(i)

#moyenne et taux de décharge des spikes à chaque temps

for i, t in enumerate(time):
  moyenne[i]=sum(mat_trials_plus5[0:iter,i])/iter

i=0
j=0
while i <= len(time):
  rate_plus5[j]=sum(moyenne[i:i+a])/(a*dt)
  i+=a
  j+=1

##fit de la courbe cos

guess_amplitude = 4.01
guess_phase = 0.24*pi
p0=[guess_amplitude, guess_phase]

# create the function we want to fit
def fit_plus5(x, amplitude, phase):
    return cos(x * omega + phase) * amplitude + mean(rate_plus5)

fit__plus5 = curve_fit(fit_plus5, time2, rate_plus5, p0=p0)
data_first_guess = fit_plus5(time2, *p0)
data_fit_plus5 = fit_plus5(time2, *fit__plus5[0])

#########Phase et amplitude (D, E) 
Ki_Ke=[0.05,0.2,0.4,0.6,0.8,1,1.2,1.4,1.55]
Ki_Ke=asarray(Ki_Ke)

while(fit_0[0][1]>pi) :
  fit_0[0][1]-=2*pi
while(fit_0[0][1]<-1*pi) :
  fit_0[0][1]+=2*pi

while(fit__plus1[0][1]>pi) :
  fit__plus1[0][1]-=2*pi
while(fit__plus1[0][1]<-1*pi) :
  fit__plus1[0][1]+=2*pi

while(fit_1[0][1]>pi) :
  fit_1[0][1]-=2*pi
while(fit_1[0][1]<-1*pi) :
  fit_1[0][1]+=2*pi

while(fit__plus2[0][1]>pi) :
  fit__plus2[0][1]-=2*pi
while(fit__plus2[0][1]<-1*pi) :
  fit__plus2[0][1]+=2*pi

while(fit_2[0][1]>pi) :
  fit_2[0][1]-=2*pi
while(fit_2[0][1]<-1*pi) :
  fit_2[0][1]+=2*pi

while(fit__plus3[0][1]>pi) :
  fit__plus3[0][1]-=2*pi
while(fit__plus3[0][1]<-1*pi) :
  fit__plus3[0][1]+=2*pi

while(fit_3[0][1]>pi) :
  fit_3[0][1]-=2*pi
while(fit_3[0][1]<-1*pi) :
  fit_3[0][1]+=2*pi

while(fit__plus4[0][1]>pi) :
  fit__plus4[0][1]-=2*pi
while(fit__plus4[0][1]<-1*pi) :
  fit__plus4[0][1]+=2*pi

while(fit__plus5[0][1]>pi) :
  fit__plus5[0][1]-=2*pi
while(fit__plus5[0][1]<-1*pi) :
  fit__plus5[0][1]+=2*pi

phi=[fit_0[0][1],fit__plus1[0][1],fit_1[0][1],fit__plus2[0][1],fit_2[0][1],fit__plus3[0][1],fit_3[0][1],fit__plus4[0][1],fit__plus5[0][1]]
modulo=[fit_0[0][0],fit__plus1[0][0],fit_1[0][0],fit__plus2[0][0],fit_2[0][0],fit__plus3[0][0],fit_3[0][0],fit__plus4[0][0],fit__plus5[0][0]]
j=complex(0,1)
omega = 60*pi                  # rad/s
K_i = arange(0,32,0.01)

K = linspace(complex(0,0),complex(0,0),len(K_i))
phaseK = zeros(len(K_i))
BalanceIE = zeros(len(K_i))
moduleK =zeros(len(K_i))
moduleK_norm =zeros(len(K_i))

for i, Ki in enumerate(K_i):
  K[i]=(1/sqrt(2*pi))*(-(Ki*tau_i)/(j*omega*tau_i+1)*exp(-j*omega*deltasyn) + (K0_e*tau_e)/(j*omega*tau_e+1))
  phaseK[i]=cmath.phase(K[i])
  moduleK[i]=abs(K[i])
  BalanceIE[i] = K_i[i]/K0_e

for i, Ki in enumerate(K_i):
  moduleK_norm[i]=moduleK[i]/moduleK[5]

modulo_norm=modulo/fit_0[0][0]


###########################################################Create a figure


fig = plt.figure()



# define sub-panel grid and possibly width and height ratios

gs0= gridspec.GridSpec(1, 3,
width_ratios=[1,2,1],wspace=0.4)


gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0],height_ratios=[1,1],hspace=0.4)
gs2= gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[2],height_ratios=[1,1],hspace=0.4)


#gridspec.GridSpec(3, 1,
#height_ratios=[1.3,0.5,1.3]
#)


# define vertical and horizontal spacing between panels

#gs.update(wspace=0.4,hspace=0.5)


# possibly change outer margins of the figure

plt.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.1)


# sub-panel enumerations

plt.figtext(0.005, 0.94, 'A',clip_on=False,color='black', weight='bold',size=22)

plt.figtext(0.005, 0.45, 'B',clip_on=False,color='black', weight='bold',size=22)

plt.figtext(0.27, 0.94, 'C',clip_on=False,color='black', weight='bold',size=22)

plt.figtext(0.7, 0.94, 'D',clip_on=False,color='black', weight='bold',size=22)

plt.figtext(0.7, 0.45, 'E',clip_on=False,color='black', weight='bold',size=22)





gssub0 = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=gs0[1],hspace=0.1,wspace=0.2,height_ratios=[1.5,2,2,2])


#########A - model

ax = plt.subplot(gs1[0])


ax.spines['top'].set_visible(False)

ax.spines['bottom'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.spines['left'].set_visible(False)

ax.yaxis.set_ticks_position('none')

ax.xaxis.set_ticks_position('none')

ax.set_xticklabels([])
ax.set_yticklabels([])

#image = mpimg.imread("C:\\Users\\Mathilde\\Documents\\Master_2\\TINS\\Papier\\model.png")
#plt.imshow(image)

################################B - synaptic filter
ax = plt.subplot(gs1[1])


ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')

ax.xaxis.set_ticks_position('bottom')

plt.xlabel(r'$t\:(ms)$')
plt.ylabel(r'$K$')
plt.xlim(0,0.05)
graduationx=[r'$0$',r'$10$',r'$20$',r'$30$',r'$40$',r'$50$']
ax.xaxis.set_ticklabels(graduationx)
plt.ylim(-20,22)
graduationy=[r'$-20$', r'$ -10 $',r'$0$',r'$10$',r'$20$']
ax.yaxis.set_ticks(arange(-20,22, 10))
ax.yaxis.set_ticklabels(graduationy)
plt.plot(time,K1,'r',linewidth=1.5)
plt.plot(time,K2,'y',linewidth=1.5)
plt.plot(time,K3,'g',linewidth=1.5)
plt.plot(time,K4,'b',linewidth=1.5)
#plt.legend()


time2=time2-.1
#####################C - Activity examples
axes = plt.gca()

#####################################################
######entries
plot0=plt.subplot(gssub0[0])
plt.plot(time,I_pre,'r',linewidth=1.5)
plt.ylim(-4E-9,4E-9)
plt.xlim(0,0.1)
graduationy=[r'$-3$', r'$ 0 $',r'$3$']
plot0.yaxis.set_ticks(arange(-3E-9,4E-9,3E-9))
plot0.yaxis.set_ticklabels(graduationy)
plot0.set_xticklabels([])
plot0.spines['top'].set_visible(False)

plot0.spines['right'].set_visible(False)

plot0.yaxis.set_ticks_position('left')

plot0.xaxis.set_ticks_position('none')

plot0.spines['bottom'].set_visible(False)
plt.ylabel(r'$I \: (nA)$')

###voltage plot
plot1_0 = plt.subplot(gssub0[4])
plt.plot(time,Vm0[130,],'k')
plt.legend(prop={'size':11})
plt.xlim(0.1,0.2)
plt.ylim(-0.17,0.100)
graduationy=[r'$-40$', r'$0$',r'$ 40 $',r'$80$',r'$120$']
plot1_0.yaxis.set_ticks(arange(-0.04,0.12, 0.04))
plot1_0.yaxis.set_ticklabels(graduationy)
plot1_0.set_xticklabels([])
plot1_0.spines['top'].set_visible(False)

plot1_0.spines['right'].set_visible(False)

plot1_0.yaxis.set_ticks_position('left')

plot1_0.xaxis.set_ticks_position('none')

plot1_0.spines['bottom'].set_visible(False)
plt.ylabel(r'$V \: (mV)$')

###raster plot
plot2_0 = plt.subplot(gssub0[8])
plt.plot(raster0[0],raster0[1],"ko",mec="k",ms=2.0,color="k")
plt.ylabel(r'$Trials$')
plot2_0.set_xticklabels([])
plt.ylim(0,350)
plt.xlim(0,len(time)/2)
plot2_0.yaxis.set_ticks(arange(0,400,100))
graduationy=[r'$0$',r'$100$',r'$200$',r'$300$']
plot2_0.yaxis.set_ticklabels(graduationy)
plot2_0.spines['top'].set_visible(False)

plot2_0.spines['right'].set_visible(False)

plot2_0.yaxis.set_ticks_position('left')

plot2_0.xaxis.set_ticks_position('none')

plot2_0.spines['bottom'].set_visible(False)


plot3_0 = plt.subplot(gssub0[12])
plt.bar(time2,rate0,0.002,facecolor="white",edgecolor='black')
plt.xlim(0,0.1)
plt.ylim(0,19)
graduationy=[r'$0$', r'$5$',r'$ 10 $',r'$15$',r'$20$']
plot3_0.yaxis.set_ticklabels(graduationy)
plt.ylabel(r'$Rate \:(Hz)$')
plt.plot(time2,data_fit0, 'r',linewidth=1.5)
graduationx=[r'$0$',r'$50$',r'$100$']
plot3_0.xaxis.set_ticks(arange(0,0.12,0.05))
plot3_0.xaxis.set_ticklabels(graduationx)
plot3_0.spines['top'].set_visible(False)

plot3_0.spines['right'].set_visible(False)

plot3_0.yaxis.set_ticks_position('left')

plot3_0.xaxis.set_ticks_position('bottom')

plt.legend(prop={'size':11})
plt.xlabel(r'$t \: (ms)$')


#####################################################################################
plot1=plt.subplot(gssub0[1])
plt.plot(time,I_pre,'y',linewidth=1.5)
plt.ylim(-4E-9,4E-9)
plt.xlim(0,0.1)
#graduationy=[r'$-400$', r'$-200$',r'$ 0 $',r'$200$',r'$400$']
plot1.yaxis.set_ticks(arange(-3E-9,4E-9,3E-9))
#plot1.yaxis.set_ticklabels(graduationy)
plot1.set_xticklabels([])
plot1.set_yticklabels([])
plot1.spines['top'].set_visible(False)

plot1.spines['right'].set_visible(False)

plot1.yaxis.set_ticks_position('left')

plot1.xaxis.set_ticks_position('none')

plot1.spines['bottom'].set_visible(False)
#plt.ylabel(r'$I \: (nA)$')

###voltage plot
plot1_1 = plt.subplot(gssub0[5])
plt.plot(time,Vm1[140,],'k')
plt.legend(prop={'size':11})
plt.xlim(0.1,0.2)
plt.ylim(-0.17,0.100)
plot1_1.yaxis.set_ticks(arange(-0.04,0.12, 0.04))
plot1_1.set_xticklabels([])
plot1_1.set_yticklabels([])
plot1_1.spines['top'].set_visible(False)

plot1_1.spines['right'].set_visible(False)

plot1_1.yaxis.set_ticks_position('left')

plot1_1.xaxis.set_ticks_position('none')

plot1_1.spines['bottom'].set_visible(False)



###raster plot
plot2_1 = plt.subplot(gssub0[9])
plt.plot(raster1[0],raster1[1],"ko",mec="k",ms=2.0,color="k")
plot2_1.set_yticklabels([])
plot2_1.set_xticklabels([])
plt.xlim(0,len(time)/2)
plt.ylim(0,350)
plot2_1.yaxis.set_ticks(arange(0,400,100))
plot2_1.spines['top'].set_visible(False)

plot2_1.spines['right'].set_visible(False)

plot2_1.yaxis.set_ticks_position('left')

plot2_1.xaxis.set_ticks_position('none')

plot2_1.spines['bottom'].set_visible(False)



plot3_1 = plt.subplot(gssub0[13])
plt.bar(time2,rate1,0.002,facecolor="white",edgecolor='black')
plt.xlim(0,0.1)
plt.ylim(0,19)
plt.plot(time2,data_fit1, 'y',linewidth=1.5)
plot3_1.set_yticklabels([])
graduationx=[r'$0$',r'$50$',r'$100$']
plot3_1.xaxis.set_ticks(arange(0,0.12,0.05))
plot3_1.xaxis.set_ticklabels(graduationx)
plt.legend(prop={'size':11})
plt.xlabel(r'$t \: (ms)$')
plot3_1.spines['top'].set_visible(False)

plot3_1.spines['right'].set_visible(False)

plot3_1.yaxis.set_ticks_position('left')

plot3_1.xaxis.set_ticks_position('bottom')


#######################################################################################
plot2=plt.subplot(gssub0[2])
plt.plot(time,I_pre,'g',linewidth=1.5)
plt.ylim(-4E-9,4E-9)
plt.xlim(0,0.1)
#graduationy=[r'$-400$', r'$-200$',r'$ 0 $',r'$200$',r'$400$']
plot2.set_yticklabels([])
plot2.yaxis.set_ticks(arange(-3E-9,4E-9,3E-9))
#plot2.yaxis.set_ticklabels(graduationy)
plot2.set_xticklabels([])
plot2.spines['top'].set_visible(False)

plot2.spines['right'].set_visible(False)

plot2.yaxis.set_ticks_position('left')

plot2.xaxis.set_ticks_position('none')

plot2.spines['bottom'].set_visible(False)
#plt.ylabel(r'$I \: (nA)$')

###voltage plot
plot1_2 = plt.subplot(gssub0[6])
plt.plot(time,Vm2[150,],'k')
plt.legend(prop={'size':11})
plt.xlim(0.1,0.2)
plt.ylim(-0.17,0.100)
plot1_2.set_xticklabels([])
plot1_2.set_yticklabels([])
#graduationy=[r'$-40$', r'$0$',r'$ 40 $',r'$80$',r'$120$']
plot1_2.yaxis.set_ticks(arange(-0.04,0.12, 0.04))
#plot1_2.yaxis.set_ticklabels(graduationy)
plot1_2.spines['top'].set_visible(False)

plot1_2.spines['right'].set_visible(False)

plot1_2.yaxis.set_ticks_position('left')

plot1_2.xaxis.set_ticks_position('none')

plot1_2.spines['bottom'].set_visible(False)



###raster plot
plot2_2 = plt.subplot(gssub0[10])
plt.ylim(0,350)
plt.xlim(0,len(time)/2)
plt.plot(raster2[0],raster2[1],"ko",mec="k",ms=2.0,color="k")
plot2_2.set_xticklabels([])
plot2_2.yaxis.set_ticks(arange(0,400,100))
plot2_2.set_yticklabels([])
#plt.ylabel(r'$Trials$')
#graduationy=[r'$0$', r'$80$',r'$ 160 $',r'$240$',r'$320$']
#plot2_2.yaxis.set_ticklabels(graduationy)
plot2_2.spines['top'].set_visible(False)

plot2_2.spines['right'].set_visible(False)

plot2_2.yaxis.set_ticks_position('left')

plot2_2.xaxis.set_ticks_position('none')

plot2_2.spines['bottom'].set_visible(False)



plot3_2 = plt.subplot(gssub0[14])
plt.bar(time2,rate2,0.002,facecolor="white",edgecolor='black')
plt.xlim(0,0.1)
plt.plot(time2,data_fit2, 'g',linewidth=1.5)
#graduationy=[r'$0$', r'$5$',r'$ 10 $',r'$15$',r'$20$']
#plot3_2.yaxis.set_ticklabels(graduationy)
graduationx=[r'$0$',r'$50$',r'$100$']
plot3_2.xaxis.set_ticks(arange(0,0.12,0.05))
plot3_2.xaxis.set_ticklabels(graduationx)
plot3_2.set_yticklabels([])
plt.legend(prop={'size':11})
plt.xlabel(r'$t \: (ms)$')
plt.ylim(0,19)
plot3_2.spines['top'].set_visible(False)

plot3_2.spines['right'].set_visible(False)

plot3_2.yaxis.set_ticks_position('left')

plot3_2.xaxis.set_ticks_position('bottom')


###################################
plot3=plt.subplot(gssub0[3])
plt.plot(time,I_pre,'b',linewidth=1.5)
plt.ylim(-4E-9,4E-9)
plt.xlim(0,0.1)
#graduationy=[r'$-400$', r'$-200$',r'$ 0 $',r'$200$',r'$400$']
plot3.yaxis.set_ticks(arange(-3E-9,4E-9,3E-9))
#plot3.yaxis.set_ticklabels(graduationy)
plot3.set_xticklabels([])
plot3.set_yticklabels([])
plot3.spines['top'].set_visible(False)

plot3.spines['right'].set_visible(False)

plot3.yaxis.set_ticks_position('left')

plot3.xaxis.set_ticks_position('none')

plot3.spines['bottom'].set_visible(False)
#plt.ylabel(r'$I \: (nA)$')

###voltage plot
plot1_3 = plt.subplot(gssub0[7])
plt.plot(time,Vm3[160,],'k')
plt.legend(prop={'size':11})
plt.xlim(0.1,0.2)
plt.ylim(-0.17,0.100)
plot1_3.set_xticklabels([])
plot1_3.set_yticklabels([])
plot1_3.yaxis.set_ticks(arange(-0.04,0.12, 0.04))
plot1_3.spines['top'].set_visible(False)

plot1_3.spines['right'].set_visible(False)

plot1_3.yaxis.set_ticks_position('left')

plot1_3.xaxis.set_ticks_position('none')

plot1_3.spines['bottom'].set_visible(False)



###raster plot
plot2_3 = plt.subplot(gssub0[11])
plt.ylim(0,350)
plt.xlim(0,len(time)/2)
plt.plot(raster3[0],raster3[1],"ko",mec="k",ms=2.0,color="k")
plot2_3.set_xticklabels([])
plot2_3.yaxis.set_ticks(arange(0,400,100))
plot2_3.set_yticklabels([])
plot2_3.spines['top'].set_visible(False)

plot2_3.spines['right'].set_visible(False)

plot2_3.yaxis.set_ticks_position('left')

plot2_3.xaxis.set_ticks_position('none')

plot2_3.spines['bottom'].set_visible(False)


plot3_3 = plt.subplot(gssub0[15])
plt.bar(time2,rate3,0.002,facecolor="white",edgecolor='black')
plt.xlim(0,0.1)
plt.plot(time2,data_fit3, 'b',linewidth=1.5)
plot3_3.set_yticklabels([])
graduationx=[r'$0$',r'$50$',r'$100$']
plot3_3.xaxis.set_ticks(arange(0,0.12,0.05))
plot3_3.xaxis.set_ticklabels(graduationx)
plt.legend(prop={'size':11})
plt.xlabel(r'$t \: (ms)$')
plt.ylim(0,19)
plot3_3.spines['top'].set_visible(False)

plot3_3.spines['right'].set_visible(False)

plot3_3.yaxis.set_ticks_position('left')

plot3_3.xaxis.set_ticks_position('bottom')

#time2=time2+.1

######################D - Phase
ax=plt.subplot(gs2[0])
plt.xlim(0,1.6)
#plt.ylim(-0.45*pi,0.33*pi)
plt.plot(BalanceIE,phaseK-(0.2*pi),'k')
plt.legend(loc="upper left")
plt.xlabel(r'$K_i/K_e$')
plt.ylabel(r'$Output Phase (rad)$')
ax.yaxis.set_ticks(arange(-0.45*pi,0.31*pi, 0.15*pi))
graduationy=[r'$-0.45 \pi$',r'$-0.30 \pi$',r'$-0.15 \pi$', r'$0$', r'$ 0.15\pi $',r'$ 0.30 \pi $']
ax.yaxis.set_ticklabels(graduationy)
ax.xaxis.set_ticks(arange(0,1.8,0.4))
graduationx=[r'$0$',r'$0.4$',r'$0.8$',r'$1.2$',r'$1.6$']
ax.xaxis.set_ticklabels(graduationx)
plt.plot(Ki_Ke,phi,'ko')
ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')

ax.xaxis.set_ticks_position('bottom')


######################E - Amplitude
ax=plt.subplot(gs2[1])
plt.plot(BalanceIE,0.5*moduleK_norm,'k')
plt.xlim(0,1.6)
plt.xlabel(r'$K_i/K_e$')
plt.ylim(0,7)
plt.ylabel(r'$Output Amplitude (Hz)$')
ax.yaxis.set_ticks(arange(0,7, 2))
#graduationy=[r'$0$', r'$ 2 $',r'$4 $', r'$6$',r'$8$']
#ax.yaxis.set_ticklabels(graduationy)
ax.xaxis.set_ticks(arange(0,2,0.4))
graduationx=[r'$0$',r'$0.4$',r'$0.8$',r'$1.2$',r'$1.6$',r'$1.8$']
ax.xaxis.set_ticklabels(graduationx)
plt.plot(Ki_Ke,0.5*modulo_norm,'ko')
ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)

ax.yaxis.set_ticks_position('left')

ax.xaxis.set_ticks_position('bottom')


plt.savefig("high_noise2.pdf")

plt.show()





