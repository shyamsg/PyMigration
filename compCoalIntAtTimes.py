import numpy as np

def comp_pw_coal_cont(m, Ne_inv):
    """Function to evaluate the coal. intensities for the given 
    migration and effective population sizes, in the time window 
    of t generations. Here we use compute an infinitesimal rate 
    matrix Q from the migration rates and population sizes and use
    the exponentiation of this rate matrix to compute the transition
    probabilities.
    """
    numdemes = len(Ne_inv)
    if np.shape(m) != (numdemes, numdemes):
        print np.shape(m)
        print numdemes
        raise Exception('Migration matrix and population size vector indicate different number of demes.')
    nr = numdemes * (numdemes + 1) / 2 + 1
    Q = np.zeros((nr, nr))
    rnum = -1
    demePairs = {}
    for p1 in xrange(numdemes):
        for p2 in xrange(p1, numdemes):
            rnum += 1
            demePairs[p1, p2] = rnum

    for dp1 in demePairs.keys():
        if dp1[0] == dp1[1]:
            Q[demePairs[dp1]][nr - 1] = 0.5 * Ne_inv[dp1[0]]
        for dp2 in demePairs.keys():
            if dp1 == dp2:
                continue
            if dp1[0] == dp1[1]:
                if dp2[0] == dp2[1]:
                    continue
                elif dp2[0] == dp1[0]:
                    Q[demePairs[dp1]][demePairs[dp2]] = 2 * m[dp1[1]][dp2[1]]
                elif dp2[1] == dp1[1]:
                    Q[demePairs[dp1]][demePairs[dp2]] = 2 * m[dp1[0]][dp2[0]]
            elif dp2[0] in dp1:
                if dp1[0] == dp2[0]:
                    Q[demePairs[dp1]][demePairs[dp2]] = m[dp1[1]][dp2[1]]
                else:
                    Q[demePairs[dp1]][demePairs[dp2]] = m[dp1[0]][dp2[1]]
            elif dp2[1] in dp1:
                if dp1[0] == dp2[1]:
                    Q[demePairs[dp1]][demePairs[dp2]] = m[dp1[1]][dp2[0]]
                else:
                    Q[demePairs[dp1]][demePairs[dp2]] = m[dp1[0]][dp2[0]]

    for dp1 in demePairs.keys():
        Q[demePairs[dp1]][demePairs[dp1]] = -sum(Q[demePairs[dp1]][:])

    return Q

def expM(Q):
    """Computes the matrix exponential using an eigenvalue decomposition
    """
#    if not hasattr(expM, "cnt"):
#        expM.cnt = 0
    COND_NUM_THRES = 1e+50
    EIG_THRES = 1e-20
    L, U = linalg.eig(Q)
    slm = np.max(np.abs(L))
    for i in xrange(len(L)):
        if abs(L[i] / slm) < EIG_THRES:
            L[i] = 0

    L = np.diag(np.exp(L))
    U = np.matrix(U)
    eQ = U * L * U.I
    return eQ

def conv_scrambling_matrix(P):
    """This takes a Probability matrix and converts it to a scrambling matrix,
    to get the probability of lines moving from the current demes to any config
    of demes at the beginning of the time slice.
    """
    Pnew = np.matrix(P.copy())
    sz = np.shape(Pnew)
    Pnew[0:sz[0] - 1, sz[1] - 1] = 0
    Prowsum = np.matrix(np.sum(Pnew, 1))
    Pnew = Pnew / Prowsum
    return Pnew

def converge_pops(popDict, scramb):
    """Changes the scrambling matrix appropriately upon
    the merging of populations.
    """
    npop = np.shape(scramb)[1]
    npop = int(np.real(np.sqrt(8 * npop - 7)) / 2)
    nr = np.shape(scramb)[0]
    nc = len(popDict)
    nc = nc * (nc + 1) / 2 + 1
    temp = np.matrix(np.zeros((nr, nc)))
    temp[:, -1] = np.real(scramb[:, -1])
    ptc = pop_to_col(popDict, npop)
    for j in xrange(nc - 1):
        temp[:, j] = np.real(np.sum(scramb[:, ptc[j]], 1))
    return temp

def pop_to_col(popD, nd_old):
    """Given the population indices, it tells us
    the correspoiding columns where lines come
    from any of the pops in the list
    """
    nd = len(popD)
    ptc = []
    for i in range(nd):
        for j in range(i, nd):
            cols = []
            for a in popD[i]:
                for b in popD[j]:
                    if a > b and i == j:
                        continue
                    if a <= b:
                        tt = a * (2 * nd_old - a + 1) / 2 + b - a
                    elif b < a:
                        tt = b * (2 * nd_old - b + 1) / 2 + a - b
                    if tt not in cols:
                        cols.append(tt)
            ptc.append(cols)
    return ptc

def compute_pw_coal_rates(Nes, ms, ts, popmaps):
    """Given a list of population sizes and the migration matrix,
    one for each time slice, compute the expected coalescent rates
    for all possible pairs of lines. Uses the comp_pw_coal_cont 
    function to do this for each time slice.
    The lists ms and Nes must be ordered from most recent to most 
    ancient. Here ts is the list of time slice lengths - in the same
    order. popmaps is a list which tells which pops merge back 
    in time at the beginning of this time slice. Each entry
    of popmaps is a list of arrays or tuples with all pops that 
    merge into one
    """
    numslices = len(ts)
    numpops = len(Nes[0])
    nr = numpops * (numpops + 1) / 2 + 1
    exp_rates = np.matrix(np.zeros((nr - 1, numslices)))
    P0 = np.matrix(np.eye(nr))
    for i in xrange(numslices):
        Ne_inv = [ 1.0 / x for x in Nes[i] ]
        mtemp = ms[i]
        oldnumpops = numpops
        numpops = len(Ne_inv)
        m = np.zeros((numpops, numpops))
        cnt = 0
        if len(popmaps[i]) == numpops:
            P0 = converge_pops(popmaps[i], P0)
        elif len(popmaps[i]) > 0:
            print len(popmaps[i])
            print numpops
            raise Exception('Population map and other parameters do not match ' + str(i))
        for ii in xrange(numpops):
            for jj in xrange(ii + 1, numpops):
                m[ii, jj] = m[jj, ii] = mtemp[cnt]
                cnt += 1
        Q = comp_pw_coal_cont(m, Ne_inv)
        eQ = expM(ts[i] * Q)
        P = P0 * eQ
        exp_rates[:, i] = np.real(P[0:nr - 1, -1])
        P0 = P0 * conv_scrambling_matrix(eQ)
    exp_rates = np.matrix(np.max((np.zeros(np.shape(exp_rates)), exp_rates), 0))
    return exp_rates

# This is an example for using this piece of code to compute the 
# item. First I will use an example with 2 populations

########################################################
# Population demography - 2 populations - merge back   #
# 300 generations ago. The times are split into 50 gens#
# Migration rates change from 1e-5 to 5e-5 to 1e-4.    #
# We look at 8 total slices, 6 before merging and 2    #
# after.                                               #
########################################################

Ns = [[30000, 30000], [20000, 10000], [10000, 10000], [10000, 10000],
      [10000, 10000], [10000, 10000], [10000], [10000]]
ms = [[0], [0], [1e-5], [5e-5], [5e-5], [1e-4], [], []]
ts = [50, 50, 50, 50, 50, 50, 50, 50]
pd = [[], [], [], [], [], [], [(0,1)], []]

coal_rates = compute_pw_coal_rates(Ns, ms, ts, pd)

########################################################
# Population demography - 3 populations - (1,2) merge  #
# 100 generations ago. The other merge is at 300 gens  #
# The times are split into 50 gens each slice. Conti-  #
# ous migration between 1 and 2, none from 0 to 1 or 2 #
# Migration rates change from 1e-5 to 5e-5 to 1e-4 for #
# the migration from 0 to (1,2).                       #
# There are 4 slices 8 slices in total. We look at 8   # 
# total slices, 6 before merging and 2 after.          #
########################################################

Ns = [[20000, 30000, 30000], [20000, 20000, 10000], [10000, 15000], [10000, 15000],
      [10000, 10000], [10000, 10000], [10000], [10000]]
ms = [[0, 0, 7e-5], [0, 0, 1e-4], [1e-5], [5e-5], [5e-5], [1e-4], [], []]
ts = [50, 50, 50, 50, 50, 50, 50, 50]
pd = [[], [], [(0,), (1,2)], [], [], [], [(0,1)], []]

coal_rates = compute_pw_coal_rates(Ns, ms, ts, pd)

