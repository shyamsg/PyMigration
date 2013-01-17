#Embedded file name: mkGraph.py
import numpy as np
import pickle
fl = open('SimKnown_1_rep100.pck')
xd, ed = pickle.load(fl)
fl.close()
edmeans = []
edvars = []
nl = [0.001,
 0.005,
 0.01,
 0.02,
 0.05,
 0.1,
 0.2,
 0.5]
for noise in nl:
    ted = ed[str(noise)]
    edmeans.append(np.mean(ted, 0))
    edvars.append(np.var(ted, 0))

edsd = np.sqrt(edvars)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
f = plt.figure()
ax = f.add_subplot(111)
segs = np.zeros((8, 7, 2), float)
segs[:, :, 1] = np.array(edmeans)
from matplotlib.colors import colorConverter
for i in xrange(8):
    segs[i, :, 0] = np.array([0,
     1,
     2,
     3,
     4,
     5,
     6])

linesegs = LineCollection(segs, colors=[ colorConverter.to_rgba(i) for i in ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'b') ])
ax.add_collection(linesegs, label=[ str(x) for x in nl ])
ax.set_xlim((0, 7))
ax.set_ylim((0, 0.5))
ax.set_ylabel('Maximum relative error in the estimates')
ax.set_xlabel('Time Slice')
plt.legend()
plt.show()