
# coding: utf-8

# In[15]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

# the animation modules from matplotlib

import matplotlib.animation as animation
import matplotlib.patches as patches

import sys
import os

sys.path.append('./modules')
#sys.path.append("/Users/jonathancampbell/Documents/Data Mining/data mining final")
from importlib import reload
import graphDist
import notepicker as npkr
reload(npkr)

# ===========================================================
# READ IN THE DATA FILE ! ! !
#breath = np.loadtxt("./data/breath_CW2.txt")

#os.chdir("/Users/jonathancampbell/Documents/upload/2_data/data_examples/")
#breath = np.loadtxt("breath_BH_y1701_1.txt")
#pwd()
df = pd.read_csv("Rdf.csv")

X = df["X1"]
Y = df["X2"]
col = df["fit.cluster"]

df = df.sort_values(["X1"])
df = df.reset_index(drop = True)
df["index"] = df.index + 1

print(df)


df
#print(df.columns)
time = df["index"]
#time = np.asarray(df[df.columns[0]])[:300]
#temp_C = max(abs(df["9-MFCCs1m"])) + df["9-MFCCs1m"]
#temp_C = temp_C[:300]
#humidity = df["10-MFCCs2m"][:300]


dt = 1
print(dt)
print(len(time))
print(dt*len(time))

# ===========================================================
# PLOT THE DATA AS CURVES...
#fig = plt.figure(figsize=(10,8))

#plt.subplot(2,1,1)
#plt.plot(time,humidity,'b-')
#plt.xlabel('time')
#plt.ylabel('H humidity [%]')

#plt.subplot(2,1,2)
#plt.plot(time,temp_C,'r-')
#plt.xlabel('time [seconds] ')
#plt.ylabel('T temperature [C]')

#plt.show()

# ==================================================
# initialize the objects
fig = plt.figure(figsize=(6,6))
axes1 = plt.subplot(1,1,1)

axes1.scatter(X, Y, c = col)
#axes1.set_xlabel('T, temperature [C]')
#axes1.set_ylabel('H, humidity [%]')


# here we are using the class above to scale the ellipse axes to the data so they are round dots.
rad_pix = 20
dot1_width = graphDist.GraphDist(rad_pix, axes1, True)
dot1_height = graphDist.GraphDist(rad_pix, axes1, False)
patch1 = patches.Ellipse((X[0], Y[0]), dot1_width, dot1_height, fc='red')

# ===================================================
# initialize what will move in the animation
def init():
    patch1.center = (X[0], Y[0])
    axes1.add_patch(patch1)
    return patch1

# ===================================================
# define what will move in the animation
def animate(i): # is this (i) needed for the animate function
    x , y = X[i], Y[i]
    patch1.center = (x , y)
    axes1.add_patch(patch1)
    return patch1,

# ====================================================
# RUN the animation !
# flag to determine whether movie gets viewed (0) or recorded (1), not both.
view_or_write = 1
# to save, you must first install ffmpeg (and then restart the notebook engine)
path_to_save = './'
movie_name = 's1_slidingDot.mp4'
movie_dur = 7.5 # in seconds

# figure out the movie frame interval
n_frames = len(Y)
print('num frames = ' + str(n_frames))
print('movie_dur = ' + str(movie_dur))
frame_interval_sec = movie_dur/n_frames
print('frame_interval = ' + str(frame_interval_sec))
fps_ideal = 1./frame_interval_sec
print('frames per sec = ' + str(fps_ideal))

if view_or_write==0:
    interv = (1000)*(dt)/5
    print(interv)
elif view_or_write==1:
    interv = frame_interval_sec
    print(interv)

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=len(Y),
                               interval=interv) #,repeat=False)
#                               blit=True)

if view_or_write==0:
    plt.show()
elif view_or_write==1:
    anim.save(path_to_save+movie_name, fps=fps_ideal, extra_args=['-vcodec', 'h264','-pix_fmt', 'yuv420p'])


# In[16]:


# MAKE THE FIRST SET OF POSSIBLE NOTES TO CHOOSE FROM ! (chromatic scales)

# I.e the range of values that will be interpolated TO !

methodFlag = 2
# 1 = MIDI (midi cents) 
# 2 = RTcmix (frequencies)

n_octaves_total = 2.0

if methodFlag == 1: 
    root_note  = 48      
    mNotes = range(root_note, root_note+n_octaves_total*12+1) 
elif methodFlag == 2: 
    root_note  = 220.0
    top_note = root_note*(2**n_octaves_total)
    interval = 2.0**(1./12.)
    print(root_note,top_note,interval)
    #mNotes = np.arange(root_note,top_note+interval,interval) 
    # FIX THIS ! These are not correct intervals.. Musimathics
    #np.logspace(2.0, 3.0, num=4, base=2.0)
    # chromatic scale: ( but doesnt have to be... )
    mNotes = np.logspace(np.log2(root_note), np.log2(top_note), num=3*12, base=2.0)
    
print(mNotes)


# In[17]:


# define the range of data values that will be interpolated FROM 
# THEN INTERPOLATE THE DATA FROM HUMIDITY TO FREQUENCY SPACE

def interpYvals_to_freq(y):
    y_min = min(y)
    y_max = max(y) 
    # linear array of the possible data values
    y_interp = np.linspace(y_min ,y_max+(y_max-y_min)/len(mNotes), len(mNotes))
    
    data_in_freqvals = np.interp(y,y_interp,mNotes)
    
    return data_in_freqvals

data_in_freqvals = interpYvals_to_freq(df["X9.MFCCs1m"])
print(data_in_freqvals)


# In[18]:


# TIME ! what do we want the duration of the sound to be? 
#print(time[0],time[-1])

print(movie_dur)

time_movie = np.linspace(0.0,movie_dur,len(time))
print(time_movie[0],time_movie[-1])
print(len(time))
print(len(time_movie))


# In[19]:


import notepicker_vX as npkr
reload(npkr)

# MAP NOTE VALUES TO DATA
# by ... illustrate this !! 

sp = npkr.Spline(time_movie, data_in_freqvals)
values = mNotes # [60., 70., 80., 100.]
#print(values)
roots = sp.findroots(values)
print(roots.size)

notes = roots[:-1,1]
#print(notes)
times = roots[:-1,0]
durations = roots[1:,0]-roots[:-1,0]

#notes = roots[:-1,1]
#times = roots[:-1,0]
#durations = times[1:-1] - times[0:-2] 
#durations = roots[1:,0]-roots[:-1,0]

print(times[0], times[-1])
print(durations[0], durations[-1])




# In[20]:


plt.subplot(2,1,1)
plt.plot(times, notes)
#plt.xlabel('time')
plt.ylabel('Notes [freq or midi val]')
#plt.title('Humidity')
plt.subplot(2,1,2)
plt.plot(times,durations)
plt.xlabel('time [seconds] ')
plt.ylabel('durations [s]')

plt.show()


# In[21]:


# GENERATE THE RTcmix score ! (alternate to generating the midi score)
base_name = 'final'
score_name = base_name + '.sco'

# ====================
f_out = open("./" + score_name , 'w')
# YOU MUST DELETE THE SOUND FILE BEFORE RUNNING (either with python or with -clobber )
f_out.write("set_option(\"clobber = on\")")

f_out.write("rtsetparams(44100, 1)\n")
f_out.write("load(\"WAVETABLE\")\n")

output_string = 'rtoutput(\"' + base_name + '.wav\")\n'
# don't need the brackets to make it an array !
print(output_string)
f_out.write(output_string)

f_out.write("waveform = maketable(\"wave\", 1000, 1.0, 0.4, 0.2)\n")
f_out.write("ampenv = maketable(\"window\", 1000, \"hamming\")\n")
# write out the score !
# (start time, duration, amplitude, frequency, channel mix [0 left, 1.0 right],
# table_handle (which waveform to use)

# for now, constants:

# reset(44100) makes it very very smooth...

amp = 10000
mix = 0.5
tab_han = 'waveform'

for i,note_val in enumerate(notes):
    t_start = times[i]
    dur = durations[i]
    freq = note_val
    note_string = 'WAVETABLE(' + str(t_start) + ', '                   + str(dur)  + ', ' + str(amp)+ '*ampenv' + ', '                   + str(freq)  + ', ' + str(mix)  + ', '                   +  tab_han + ')\n'
    f_out.write(note_string)
f_out.close()

# ===========================


# In[22]:


#CMIX < fourth_RTcmix_algo2.sco
cmix_cmd = 'CMIX < ' + score_name
print(cmix_cmd)

#  https://blog.dominodatalab.com/lesser-known-ways-of-using-notebooks/
# %% bash

get_ipython().system(' pwd')
get_ipython().system(' ls *.sco')


# In[23]:



from subprocess import Popen

import subprocess as sp
#call(["ls", "-l"])

# this works:
ls_output = sp.check_output(['pwd'])
print(ls_output)
# but THIS works better !
#! pwd

# NOTE if the file exists already, you *MAY* have to delete it by hand to make a new one. not sure.
runCMIX = sp.Popen(cmix_cmd, shell=True) # if can only be called from a shell, use shell=True

runCMIX.wait()
print("hopefully i just wrote your sound file; is it here?")
#! ls *.wav


# In[24]:


# GENERATE THE MOVIE !
sound_name = base_name + '.wav'
movie_snd_name = 'final.avi'

#make single stereo track from two mono tracks --
# but this route is less preferable than generating one stereo track from RTcmix directly !
# https://trac.ffmpeg.org/wiki/AudioChannelManipulation

run_ffmpeg_cmd = 'ffmpeg -i ' + sound_name + ' -i ' + movie_name + ' ' + movie_snd_name
# ffmpeg -i breath_HumTemp_env_30.wav -i breathalyzer.mp4 Breathalyzer_snd.avi

make_movie = sp.Popen(run_ffmpeg_cmd, shell=True)
make_movie.wait()
print("And now did i just write your movie; is it here?")
#! ls *.avi
print("BUT beware ! it might take a while to write ! ")

