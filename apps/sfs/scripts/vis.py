
import os
import cv2
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='/data/real/')
parser.add_argument('-f', '--file', default='depth_refined.exr')

args = parser.parse_args()

filepath = os.path.join(args.dataset, args.file)
z = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

z = np.asarray(z).T
h, w = z.shape

x = np.arange(0, w)
y = np.arange(0, h)
xx, yy = np.meshgrid(x, y)
p = np.dstack([xx, yy, -z]).reshape(-1, 3)

import pptk
v = pptk.viewer(p, point_size=0.33)


#import trimesh
#v = trimesh.points.PointCloud(p)
#v.show()

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import matplotlib.tri as mtri

#fig = plt.figure()
#ax = fig.gca(projection='3d')
# ax.scatter3D(xx, yy, z, c=z, cmap='Greens') #super slow
# ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none') #memory error
#plt.show()
