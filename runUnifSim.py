#Embedded file name: runUnifSim.py
import simMig as sm
import sys
import pickle
Ns = [[8000, 8000],
 [500, 8000],
 [3000, 1000],
 [700, 700],
 [400],
 [4000],
 [2000]]
ms = [[0],
 [0],
 [2e-06],
 [2e-05],
 [0],
 [0],
 [0]]
ts = [1000] * 7
pd = [[],
 [],
 [],
 [],
 [(0, 1)],
 [],
 []]
noiselevels = [0.001,
 0.005,
 0.01,
 0.02,
 0.05,
 0.1,
 0.2,
 0.5]
reps = 100
xoptDict = {}
estErrDict = {}
for noise in noiselevels:
    xopts, estErr = sm.run_for_parms(Ns, ms, ts, pd, noise, reps)
    xoptDict[str(noise)] = xopts
    estErrDict[str(noise)] = estErr

f = open('SimKnown_1_rep100.pck', 'w')
pickle.dump((xoptDict, estErrDict), f)
f.close()