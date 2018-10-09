
# coding: utf-8
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
from shapely.geometry import Point
from shapely.ops import cascaded_union
import graphDist

# ===========================================================

df = pd.read_csv("clusterdf.csv")

df.columns = df.columns.str.replace('.', '_')

df = df.sort_values(["fit_cluster", "X1"])

df = df.reset_index(drop = True)

X = np.asarray(df.X1)
Y = np.asarray(df.X2)

col = df.fit_cluster

time = df.index

dt = time[1] - time[0]

# initialize the objects
#get indices for different clusters
c1 = df[df.fit_cluster == 1]
c2 = df[df.fit_cluster == 2]
c3 = df[df.fit_cluster == 3]
c4 = df[df.fit_cluster == 4]

size = 0.2
alpha = 0.3

def points():
    x = np.random.uniform(size = 100)
    y = np.random.uniform(size = 100)
    return x, y

x1, y1 = points()
x2, y2 = points()

polygons1 = [Point(X[i], Y[i]).buffer(size) for i in c1.index]
polygons1 = cascaded_union(polygons1)

polygons2 = [Point(X[i], Y[i]).buffer(size) for i in c2.index]
polygons2 = cascaded_union(polygons2)

polygons3 = [Point(X[i], Y[i]).buffer(size) for i in c3.index]
polygons3 = cascaded_union(polygons3)

polygons4 = [Point(X[i], Y[i]).buffer(size) for i in c4.index]
polygons4 = cascaded_union(polygons4)

fig = plt.figure(figsize=(5,5))
#fig = plt.figure(figsize=(5,5), dpi=720) #USE THIS if you want to save, rather than view video.

axes1 = fig.add_subplot(111, title = "Discriminant Coordinate K-Means Projection")

for polygon1 in polygons1:
    polygon1 = patches.Polygon(np.array(polygon1.exterior), facecolor = "red", lw = 0, alpha = alpha)
    axes1.add_patch(polygon1)

for polygon2 in polygons2:
    polygon2 = patches.Polygon(np.array(polygon2.exterior), facecolor = "black", lw = 0, alpha = alpha)
    axes1.add_patch(polygon2)

for polygon3 in polygons3:
    polygon3 = patches.Polygon(np.array(polygon3.exterior), facecolor = "blue", lw = 0, alpha = alpha)
    axes1.add_patch(polygon3)

for polygon4 in polygons4:
    polygon4 = patches.Polygon(np.array(polygon4.exterior), facecolor = "orange", lw = 0, alpha = alpha)
    axes1.add_patch(polygon4)

axes1.set_xlabel('X Coordinate')
axes1.set_ylabel('Y Coordinate')

axes1.axis(np.array([min(X), max(X), min(Y) - 5, max(Y) + 5]).round())

axes1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
axes1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


#fig3 = plt.figure(figsize=(4,2))
#axes3 = plt.subplot(1,1,1)

#axes3.scatter(df.index, df.X9_MFCCs1m)
#axes3.plot(df.index, df.X9_MFCCs1m)

#axes3.set_xlabel('Index')
#axes3.set_ylabel('MFCC Timbre Value')
#axes3.set_title('Variance in MFCC Feature')
#plt.show() #This is a diagnostic to see if there is noticeable variation in the chosen feature--
#uncomment starting from fig3 = line if you want to see it.

rad_pix = 10
#rad_pix = 60 # USE THIS if you want to save rather than view video.

dot1_width = graphDist.GraphDist(rad_pix, axes1, True)
dot1_height = graphDist.GraphDist(rad_pix, axes1, False)
patch1 = patches.Ellipse((X[0], Y[0]), dot1_width, dot1_height, fc='red')

# ===================================================
def init():
    patch1.center = (X[0], Y[0])
    axes1.add_patch(patch1)
    return patch1
# ===================================================
def animate(i):
    x , y = X[i], Y[i]
    patch1.center = (x , y)
    axes1.add_patch(patch1)
    return patch1
# ====================================================
view_or_write = 0 #change this if you want to save the video.
# to save, you must first install ffmpeg
path_to_save = './'
movie_name = 'cluster_anim.mp4'
movie_dur = 92 # in seconds

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
                               interval=interv)
if view_or_write==0:
    plt.show()
elif view_or_write==1:
    anim.save(path_to_save+movie_name, fps=fps_ideal, extra_args=['-vcodec', 'h264','-pix_fmt', 'yuv420p'])
