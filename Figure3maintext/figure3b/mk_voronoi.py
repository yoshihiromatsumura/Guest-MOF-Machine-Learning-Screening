import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.ticker import MultipleLocator

f = open('log.umap_fs_20','r')
datalist = f.readlines()
f.close()
nnn=len(datalist)
points = []
speed = []
for num in range(nnn):
    toks = datalist[num].split(' ')
    list_x = [a.rstrip().rstrip(':').rstrip(')').lstrip('(') for a in toks if a != '']
    tmp_l = [float(list_x[2]),float(list_x[3])]
    points.append(tmp_l)
    speed.append(float(list_x[1]))
points.append([-99,99])
speed.append(0.0)
points.append([99,99])
speed.append(0.0)
points.append([99,-99])
speed.append(0.0)
points.append([-99,-99])
speed.append(0.0)

# generate Voronoi tessellation
vor = Voronoi(points)
# find min/max values for normalization
minima = min(speed)
maxima = max(speed)
# normalize chosen colormap
norm = mpl.colors.Normalize(vmin=minima, vmax=maxima, clip=False)
mapper = cm.ScalarMappable(norm=norm, cmap="jet")

# plot Voronoi diagram, and fill finite regions with color mapped from speed value
voronoi_plot_2d(vor, show_points=True, show_vertices=False, 
                point_size=1, line_width=0.1)

for r in range(len(vor.point_region)):
    region = vor.regions[vor.point_region[r]]
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon), color=mapper.to_rgba(speed[r]))

ltmp = pd.read_csv('file_subst').values
labels = [str(j[0]) for j in ltmp]
for i, label in enumerate(labels):
    plt.text(points[i][0],points[i][1],label,size='x-small',color='white')

plt.xlim(-8.2, -2.2)
plt.ylim(-1.2, 3.2)
plt.gca().yaxis.set_major_locator(MultipleLocator(1.0))
plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
plt.rcParams["font.size"] = 12
plt.show()
plt.savefig("fs_vrni_x.pdf")


