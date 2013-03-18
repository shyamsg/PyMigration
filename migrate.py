#Embedded file name: /home/shyamg/projects/Migration/Code/migrate.py
import numpy as np
import pickle
from scipy import linalg
import scipy.optimize as opt
from scipy.stats import chi2
import boundNM as bnm
import sys
import mpmath as mp
#mp.dps = 24
#mp.pretty = True
import itertools as it
import pdb

def logit(p):
    """Logit function to convert a vector of probs
    to logits.
    """
    return np.log(p/(1.0-p))

def expit(a):
    """Returns inverse logit of the values.
    """
    t = np.exp(a)
    return t/(t+1.0)

def comp_pw_coal_disc(m, Ne, t):
    """Function evaluates the coalescent intensities for the 
    given migration matrix and effective population sizes, in the
    time window of t generations (unscaled coalescent units).
    """
    numpops = len(Ne)
    coalInt = np.zeros((numpops, numpops))
    currInt = np.zeros((numpops, numpops))
    for gen in xrange(1, t + 1):
        for p1 in xrange(numpops):
            coalInt[p1, p1] = 0.5 / Ne[p1] + 2 * sum(m[p1, :] * currInt[p1, :]) + (1 - 0.5 / Ne[p1] - 2 * sum(m[p1, :])) * currInt[p1, p1]
            for p2 in xrange(p1 + 1, numpops):
                coalInt[p1, p2] = sum(m[p1, :] * currInt[:, p2]) + sum(m[p2, :] * currInt[:, p1]) + (1 - sum(m[p1, :] + m[p2, :])) * currInt[p1, p2]
                coalInt[p2, p1] = coalInt[p1, p2]

        currInt = coalInt

    return coalInt


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

def compute_Frob_norm_mig(x, t, obs_coal_rates, P0, popdict, logVal = True):
    """Function to compute the order 2 norm distance between the 
    observed coalescent intensities and coalescent intensities 
    computed using the given N and m
    The input is given as x - a vector with N and m values
    P0 is the initial scrambling matrix. Note that we work with 1/N and
    m values, as preconditioning to make the problem more spherical.
    """
    k = int((np.sqrt(1 + 8 * len(x)) - 1) / 2.0)
    Ne_inv = x[0:k]
    ms = x[k:]
    m = np.zeros((k, k))
    cnt = 0
    for i in xrange(k):
        for j in xrange(i + 1, k):
            m[i, j] = m[j, i] = ms[cnt]
            cnt += 1

    Q = comp_pw_coal_cont(m, Ne_inv)
    try:
        Pcurr = (np.round(expM(t * Q), 12))
    except ValueError as v:
        print 'IN FUNC CFNM'
        print m
        print Ne_inv
        print 'OUT FUNC CFNM'
        raise v

    dims = np.shape(P0)
    Ck = np.matrix(np.eye(dims[0]))[0:dims[0] - 1, :]
    Dk = np.matrix(np.eye(dims[1]))[dims[1] - 1, :]
    Pcurr = Ck * P0 * Pcurr * Dk.T
    if np.shape(obs_coal_rates) != np.shape(Pcurr):
        print 'Dimensions: Observed -> ' + str(np.shape(obs_coal_rates)) + ' and Computed -> ' + str(np.shape(Pcurr))
        raise Exception('Dimensions of observed and computed vectors do not match.')
    try:
        obs_temp = average_coal_rates(obs_coal_rates, popdict)
        est_temp = average_coal_rates(Pcurr, popdict)
        if logVal:
#            tempo = linalg.norm(np.log(obs_temp + 1e-200) + np.log(1 - obs_temp + 1e-200) - np.log(est_temp + 1e-200) - np.log(1 - est_temp + 1e-200), 2) ** 2
            temp1 = np.log(obs_temp + 1e-200) - np.log(est_temp + 1e-200)
            if (np.isnan(temp1).any() or np.isinf(temp1).any()):
                Pcurr = Ck * P0 * mp2np(mp.expm(t*Q)) *Dk.T
                est_temp = average_coal_rates(Pcurr, popdict)
                temp1 = np.log(obs_temp + 1e-200) - np.log(est_temp + 1e-200)            
                print '1'

            tempo = linalg.norm(temp1, 2) ** 2
        else:
            tempo = linalg.norm(obs_temp - est_temp, 2) ** 2
    except:
#        print 'obsRates', obs_coal_rates
#        print 'average', average_coal_rates(obs_coal_rates, popdict)
#        print 'curr', Pcurr
#        print 'Q', Q
#        print 'P', expM(t * Q)
#        print 'P0', P0
#        print 'obs_temp: ', obs_temp
#        print 'est_temp: ', est_temp
#        print 'm: ', m
#        print 'Ne_inv: ', Ne_inv
#        print 'popdict: ', popdict
#        print 'Q: ', Q
#        print 'eQ: ', expM(t*Q)
#        print 'exp(tQ): ', mp.expm(t*Q)
#        print 'Ck: ', Ck
#        print 'Dk: ', Dk.T
#        print 'P0: ', P0
#        print 'Pcurr: ', Ck*P0*expM(t*Q)*Dk.T
#        ist = raw_input('No way jose?')
#        print np.log(obs_temp + 1e-200)
#        print np.log(1 - obs_temp + 1e-200)
#        print np.log(est_temp + 1e-200)
#        print np.log(1 - est_temp + 1e-200)
        print 'Error in tempo 3'
        raise 

    return tempo


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
        eQ = np.real(mp2np(mp.expm(ts[i] * Q)))
        P = P0 * eQ
        exp_rates[:, i] = np.real(P[0:nr - 1, -1])
        P0 = P0 * conv_scrambling_matrix(eQ)

    exp_rates = np.matrix(np.max((np.zeros(np.shape(exp_rates)), exp_rates), 0))
    return exp_rates

def construct_poparr(popdict):
    """Convert the array from comp_N_m into the desired format for 
    converge pops.
    """
    popToIndex = {}
    nd = len(popdict)
    currIndex = 0
    for ii in xrange(nd):
        if len(popdict[ii]) == 0:
            popToIndex[ii] = currIndex
            currIndex += 1
        elif ii not in popToIndex:
            popToIndex[ii] = currIndex
            for jj in popdict[ii]:
                popToIndex[jj] = popToIndex[ii]
            currIndex += 1
        elif ii in popToIndex:
            for jj in popdict[ii]:
                popToIndex[jj] = popToIndex[ii]

    pindex = 0
    for ii in xrange(currIndex):
        foundOne = False
        for jj in popToIndex:
            if popToIndex[jj] == ii:
                foundOne = True
                popToIndex[jj] = pindex
        if (foundOne): pindex += 1

    popmap = []
    for ii in xrange(pindex):
        temp = []
        for jj in popToIndex:
            if popToIndex[jj] == ii:
                temp.append(jj)
        popmap.append(tuple(temp))

    return popmap


def comp_N_m(obs_rates, t, merge_threshold, useMigration, logVal = True, verbose = False, window=0, hack=False):
    """This function estimates the N and m parameters for the various time slices. The 
    time slices are given in a vector form 't'. t specifies the length of the time slice, not
    the time from present to end of time slice (not cumulative but atomic)
    Also, the obs_rates are given, for each time slice are given in columns of the obs_rates
    matrix. Both obs_rates and time slice lengths are given from present to past.
    """
    FTOL = 1e-15
    XTOL = 1e-15
    RESTARTS = 40
    RESEED = RESTARTS / 10
    FLIMIT = 1e-15
    numslices = len(t)
    nr = np.shape(obs_rates)[0] + 1
    numdemes = int(np.real(np.sqrt(8 * nr - 7)) / 2)
    print 'Starting iterations'
    P0 = np.matrix(np.eye(nr))
    N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
    lims = [(1e-10, 0.01)] * numdemes
    m0 = np.random.uniform(1e-08, 0.001, numdemes * (numdemes - 1) / 2)
    lims += [(0, 0.1)] * (numdemes * (numdemes - 1) / 2)
    x0 = np.hstack((N0_inv, m0))
    xopts = []
    pdlist = []
    for i in xrange(numslices):
        print 'Running for slice ', i
        if i > 0:
            print 'Slice', i - 1, 'optimum', bestxopt
            x0 = bestxopt.copy()
            xopt, fun, nit, nfev, status = bnm.fmin_bound(compute_Frob_norm_mig, x0, bounds=lims, args=(t[i],
             obs_rates[:, i],
             P0,
             make_merged_pd(pdlist),
             logVal), maxfun=100000, maxiter=10000, xtol=XTOL, ftol=FTOL, full_output=True, disp=False)
            bestxopt = xopt
            bestfval = fun
        else:
            bestxopt = None
            bestfval = 1e+200
        reestimate = True
        while reestimate:
            pdslice = make_merged_pd(pdlist)
            print 'pdslice:', pdslice
            N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
            m0 = np.random.uniform(0.008, 0.001, numdemes * (numdemes - 1) / 2)
            x0 = np.hstack((N0_inv, m0))
            lims = [(1e-15, 0.01)] * numdemes
            lims += [(0, 0.1)] * (numdemes * (numdemes - 1) / 2)
            xopt, fun, nit, nfev, status = bnm.fmin_bound(compute_Frob_norm_mig, x0, bounds=lims, args=(t[i],
             obs_rates[:, i],
             P0,
             pdslice,
             logVal), maxfun=100000, maxiter=10000, xtol=XTOL, ftol=FTOL, full_output=True, disp=False)
            if fun < bestfval:
                bestfval = fun
                bestxopt = xopt
            N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
            m0 = np.random.uniform(1e-08, 0.001, numdemes * (numdemes - 1) / 2)
            x0 = np.hstack((N0_inv, m0))
            nrestarts = 0
            xopt, fun, nit, nfev, status = bnm.fmin_bound(compute_Frob_norm_mig, x0, bounds=lims, args=(t[i],
             obs_rates[:, i],
             P0,
             pdslice,
             logVal), maxfun=100000, maxiter=10000, xtol=XTOL, ftol=FTOL, full_output=True, disp=False)
            if fun < bestfval:
                bestxopt = xopt
                bestfval = fun
            if nrestarts % 10 == 0:
                print nrestarts
            while nrestarts < RESTARTS and bestfval > FLIMIT:
                if nrestarts % RESEED != 0:
                    deviation = 1 - 0.95 ** (nrestarts % RESEED)
                    N0_inv = N0_inv * np.random.uniform(1 - deviation, 1 + deviation, np.shape(N0_inv))
                    m0 = m0 * np.random.uniform(1 - deviation, 1 + deviation, np.shape(m0))
                else:
                    N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
                    m0 = np.random.uniform(1e-08, 0.001, numdemes * (numdemes - 1) / 2)
                x0 = np.hstack((N0_inv, m0))
                xopt, fun, nit, nfev, status = bnm.fmin_bound(compute_Frob_norm_mig, x0, bounds=lims, args=(t[i],
                 obs_rates[:, i],
                 P0,
                 pdslice,
                 logVal), maxfun=100000, maxiter=10000, xtol=XTOL, ftol=FTOL, full_output=True, disp=False)
                if fun < bestfval:
                    bestfval = fun
                    bestxopt = xopt
                if nrestarts % 10 == 0:
                    print nrestarts
                nrestarts += 1

            Ne_inv = bestxopt[0:numdemes]
            mtemp = bestxopt[numdemes:]
            popdict = find_pop_merges(Ne_inv, mtemp, t[i], P0, merge_threshold, useMigration, window=window, hack=hack)
            reestimate = False
            if len(popdict) < numdemes:
                print 'Merging populations and reestimating parameters:', popdict
                Ne_inv = bestxopt[0:numdemes]
                mtemp = bestxopt[numdemes:]
                m = np.zeros((numdemes, numdemes))
                cnt = 0
                for ii in xrange(numdemes):
                    for jj in xrange(ii + 1, numdemes):
                        m[ii, jj] = m[jj, ii] = mtemp[cnt]
                        cnt += 1

                Qtemp = comp_pw_coal_cont(m, Ne_inv)
                Ptemp = expM(t[i] * Qtemp)
                if verbose:
                    print np.real(P0 * Ptemp)[0:-1, -1]
                    print np.real(obs_rates[:, i])
                    ist = raw_input('What the duece?')
                print bestxopt
                P0 = converge_pops(popdict, P0)
                reestimate = True
                pdlist.append(popdict)
                numdemes = len(popdict)
                nr = numdemes * (numdemes + 1) / 2 + 1
                lims[:] = []
                bestfval = 1e+200
                bestxopt = None
            else:
                modXopt = bestxopt[0:nr - 1].copy()
                for iii in xrange(numdemes):
                    modXopt[iii] = 1.0 / modXopt[iii]

                xopts.append(modXopt)
                Ne_inv = bestxopt[0:numdemes]
                mtemp = bestxopt[numdemes:]

        m = np.zeros((numdemes, numdemes))
        cnt = 0
        for ii in xrange(numdemes):
            for jj in xrange(ii + 1, numdemes):
                m[ii, jj] = m[jj, ii] = mtemp[cnt]
                cnt += 1

        Q = comp_pw_coal_cont(m, Ne_inv)
        P = expM(t[i] * Q)
        if verbose:
            print np.real(P0 * P)[0:-1, -1]
            print np.real(obs_rates[:, i])
            ist = raw_input('Waiting for input...')
        P0 = conv_scrambling_matrix(P0 * P)

    find_pop_merges.states_1 = None
    find_pop_merges.states_1 = None
    return (xopts, pdlist)


def cond_to_psmc(condRates):
    """Function to convert the conditional rates
    to the marginals, of the form returned by psmc.
    """
    condRates = np.array(condRates)
    marg_rates = np.zeros(np.shape(condRates))
    nr = np.shape(condRates)[0]
    nc = np.shape(condRates)[1]
    raise nr > nc or AssertionError
    marg_rates[0, :] = condRates[0, :]
    for i in xrange(1, nr):
        marg_rates[i, :] = condRates[i, :] * (1.0 - np.sum(marg_rates[0:i, :], 0))

    return marg_rates

def find_pop_merges(Ninv, mtemp, t, P0, merge_threshold, useMigration, window=0, hack=False):
    """This function takes the optimal paramters found and 
    figures out if the populations need to be merged.
    """
    numdemes = len(Ninv)
    #The following sets up the static vars that we need
    # to keep track of the previous 2 rate vectors.
    # This rates_1 is the one back rate vector
    # also need to store the 2 back scrambling
    # matrix - let us see where to store it.
    # This is not the place since this function should
    # know nothing about the outside world.
    # The resetting parts should be the duty of the 
    # calling function.
    if not hasattr(find_pop_merges, "states_1"):
        find_pop_merges.states_1 = None
        #This rates_2 is the 2 back rate vector
    if not hasattr(find_pop_merges, "states_2"):
        find_pop_merges.states_2 = None

    if (window == 2):
        print 'Using windowing scheme'
        m = np.zeros((numdemes, numdemes))
        cnt = 0
        for ii in xrange(numdemes):
            for jj in xrange(ii + 1, numdemes):
                m[ii, jj] = m[jj, ii] = mtemp[cnt]
                cnt += 1
        Q = comp_pw_coal_cont(m, Ninv)
        P = expM(t * Q)[0:-1, -1]
        popdict = []
        for i in xrange(numdemes):
            popdict.append([])

        # This is important - if the prev rates are not available, 
        # then cannot check for merges. So return no merges.
        # The merger check always check for mergers in currtime - window
        # So, if we are merging now (and window is 2), it means that we 
        # are checking for mergers 2 timeslots back. Essentially we are
        # peeking into the future, then setting mergers and resuming 
        # by resetting our pop model 2 timeslots back in the past.
        if find_pop_merges.states_1 == None:
            find_pop_merges.states_1 = (P0, P)
            return (find_pop_merges.states_2, construct_poparr(popdict))
        if find_pop_merges.states_2 == None:
            find_pop_merges.states_2 = find_pop_merges.states_1
            find_pop_merges.states_1 = (P0, P)
            return (find_pop_merges.states_2, construct_poparr(popdict))
        # The pop merger check is a 3 sample paired t-test equivalent
        # r11.1 r12.1 r22.1 are the rates in the first time slice (2 back)
        # r11.2 r12.2 r22.2 are the rates in the first time slice (1 back)
        # r11.3 r12.3 r22.3 are the rates in the first time slice (current)
        # the stupid hacky way to do this is to compute the sd/mean in each 
        # timeslice - then compute the mean of these and compare this to a 
        # prespecified threshold
        sdRates = np.zeros(3)
        meanRates = np.zeros(3)
        medianRates = np.zeros(3)
        rangeRates = np.zeros(3)
        stats = np.zeros(3)
        merge = False
        for i in xrange(numdemes):
            for j in xrange(i + 1, numdemes):
                merge = False
                overdists = np.zeros((3,3))
                dists = []
                dists.append(P[(2 * numdemes - i + 1) * i / 2])
                dists.append(P[(2 * numdemes - i + 1) * i / 2 + (j - i)])
                dists.append(P[(2 * numdemes - j + 1) * j / 2])
                dists = np.real(np.array(dists).flatten())
                dists = logit(dists)
                overdists[0,:] = dists
                meanRates[0] = np.mean(dists)
                medianRates[0] = np.median(dists)
                rangeRates[0] = np.max(dists) - np.min(dists)
                sdRates[0] = np.sqrt(np.var(dists))
                dists = []
                dists.append(find_pop_merges.states_1[1][(2 * numdemes - i + 1) * i / 2])
                dists.append(find_pop_merges.states_1[1][(2 * numdemes - i + 1) * i / 2 + (j - i)])
                dists.append(find_pop_merges.states_1[1][(2 * numdemes - j + 1) * j / 2])
                dists = np.real(np.array(dists).flatten())
                dists = logit(dists)
                overdists[1,:] = dists
                meanRates[1] = np.mean(dists)
                medianRates[1] = np.median(dists)
                rangeRates[1] = np.max(dists) - np.min(dists)
                sdRates[1] = np.sqrt(np.var(dists))
                dists = []
                dists.append(find_pop_merges.states_2[1][(2 * numdemes - i + 1) * i / 2])
                dists.append(find_pop_merges.states_2[1][(2 * numdemes - i + 1) * i / 2 + (j - i)])
                dists.append(find_pop_merges.states_2[1][(2 * numdemes - j + 1) * j / 2])
                dists = np.real(np.array(dists).flatten())
                dists = logit(dists)
                overdists[2,:] = dists
                meanRates[2] = np.mean(dists)
                medianRates[2] = np.median(dists)
                rangeRates[2] = np.max(dists) - np.min(dists)
                sdRates[2] = np.sqrt(np.var(dists))
                if (hack):
                    stats = sdRates/meanRates
                    if np.mean(stats) < merge_threshold:
                        merge = True
                    else:
                        merge = False
                else:
                    # things to do. make the model for the check and
                    # then compare the rates under the model - 
                    # using a LRT to check if the non-merger
                    # model is better than the merger model
                    y_0 = overdists[:,1]-overdists[:,0]
                    y_2 = overdists[:,2]-overdists[:,0]
                    sig2_a = np.sum(y_0**2 + y_2**2)/6.0
                    m_0 = np.mean(y_0)
                    m_2 = np.mean(y_2)
                    sig2_f = np.sum((y_0-m_0)**2 + (y_2-m_2)**2)/6
                    lalt = -3*np.log(sig2_a) - 3
                    lfull = -3*np.log(sig2_f) - 3
                    lr = 2*(lfull - lalt)
#                    print find_pop_merges.states_1[1]
#                    print find_pop_merges.states_2[1]
#                    print P
#                    print overdists
                    df = 2
                    test = 0
                    if test == 1:
                        y = y_0+y_2
                        m = np.mean(y)
                        sig2_f = np.mean((y-m)**2)
                        sig2_a = np.mean(y**2)
                        lr = -3*(np.log(sig2_f)-np.log(sig2_a))
                        df = 1
                    elif test == 2:
                        
                    print y_0
                    print y_2
#                    print y, m
                    print sig2_a, sig2_f
                    print 'Likelihood ratio', lr
                    if lr > chi2.isf(merge_threshold, df): # full model fits much better than alt
                        merge = False
                    else:
                        merge = True
                if merge:
                    popdict[i].append(j)
                    popdict[j].append(i)
        state2 = find_pop_merges.states_2
        find_pop_merges.states_2 = find_pop_merges.states_1
        find_pop_merges.states_1 = (P0, P)
    elif useMigration == False:
        print 'Using coalescent rates'
        m = np.zeros((numdemes, numdemes))
        cnt = 0
        for ii in xrange(numdemes):
            for jj in xrange(ii + 1, numdemes):
                m[ii, jj] = m[jj, ii] = mtemp[cnt]
                cnt += 1
        Q = comp_pw_coal_cont(m, Ninv)
        P = expM(t * Q)[0:-1, -1]
        popdict = []
        for i in xrange(numdemes):
            popdict.append([])
        for i in xrange(numdemes):
            for j in xrange(i + 1, numdemes):
                dists = []
                dists.append(P[(2 * numdemes - i + 1) * i / 2])
                dists.append(P[(2 * numdemes - i + 1) * i / 2 + (j - i)])
                dists.append(P[(2 * numdemes - j + 1) * j / 2])
                dists = np.real(np.array(dists).flatten())
                meanRates = np.mean(dists)
                medianRates = np.median(dists)
                rangeRates = np.max(dists) - np.min(dists)
                sdRates = np.sqrt(np.var(dists))
                print 'CV:', sdRates, meanRates, rangeRates, rangeRates / meanRates, sdRates/meanRates
                if sdRates/meanRates < merge_threshold:
                    popdict[i].append(j)
                    popdict[j].append(i)
    else:
        popdict = []
        for kk in xrange(numdemes):
            popdict.append([])

        cnt = 0
        for ii in xrange(numdemes):
            for jj in xrange(ii + 1, numdemes):
                if mtemp[cnt] > merge_threshold:
                    popdict[ii].append(jj)
                    popdict[jj].append(ii)
                cnt += 1

    tellmenoe = construct_poparr(popdict)
    if len(tellmenoe) < numdemes:
        find_pop_merges.states_1 = None
        find_pop_merges.states_2 = None
    return (state2, tellmenoe)


def average_coal_rates(origrates, popdict):
    """This function changes the rates to average them
    when the populations merge. popdictlist is a list
    of population dictionaries. rates is the column vector
    of coalescent rates.
    When applying to observed rates, the popdictlist must just 
    contain the latest list. While applying to the estimated 
    rates from the current N and m, the whole list must be given.
    """
    if len(popdict) == 0:
        return origrates
    rates = origrates.copy()
    numdemes = reduce(lambda x, y: x + len(y), popdict, 0)
    ptc = pop_to_col(popdict, numdemes)
    newrates = np.zeros((len(ptc), np.shape(origrates)[1]))
    for row in xrange(len(ptc)):
        newrates[row] = np.mean(np.real(rates[ptc[row], :]), 0)
    return newrates

def make_merged_pd(pdlist):
    """Merges the sequential popdicts
    """
    if len(pdlist) == 0:
        return []
    if len(pdlist) == 1:
        return pdlist[0]
    pdnew = pdlist[-1]
    rg = range(1, len(pdlist))
    rg.reverse()
    for i in rg:
        pdprev = pdlist[i - 1]
        pdtemp = []
        for pops in pdnew:
            temp = []
            for pop in pops:
                temp = np.hstack((temp, pdprev[pop]))

            temp = [ int(x) for x in temp ]
            pdtemp.append(tuple(np.sort(temp)))

        pdnew = pdtemp

    return pdnew


def comp_N_m_bfgs(obs_rates, t, merge_threshold, useMigration, initialize = False, logVal = True, verbose = False, window=0, hack=False):
    """This function estimates the N and m parameters for the various time slices. The 
    time slices are given in a vector form 't'. t specifies the length of the time slice, not
    the time from present to end of time slice (not cumulative but atomic)
    Also, the obs_rates are given, for each time slice are given in columns of the obs_rates
    matrix. Both obs_rates and time slice lengths are given from present to past.
    """
    FTOL = 10.0
    EPSILON = 1e-13
    RESTARTS = 40
    FLIMIT = 1e-15
    numslices = len(t)
    nr = np.shape(obs_rates)[0] + 1
    numdemes = int(np.real(np.sqrt(8 * nr - 7)) / 2)
    print 'Starting iterations'
    P0 = np.matrix(np.eye(nr))
    xopts = []
    pdlist = []
#    for i in range(numslices):
    i = 0
    while i < numslices:
        print 'Running for slice ', i
        if i > 0:
            x0 = bestxopt.copy()
            xopt, fun, dic = opt.fmin_l_bfgs_b(compute_Frob_norm_mig, x0, bounds=lims, args=(t[i],
             obs_rates[:, i], P0, make_merged_pd(pdlist), logVal), approx_grad=True, maxfun=100000, 
             factr=FTOL, epsilon=EPSILON, disp=0)
            bestxopt = xopt
            bestfval = fun
        else:
            bestxopt = None
            bestfval = 1e+200
        reestimate = True
        while reestimate:
            pdslice = make_merged_pd(pdlist)
            print 'restarting in slice', i
            if initialize:
                x0 = initStartingPoint(obs_rates[:, i], t[i], make_merged_pd(pdslice), P0)
            else:
                N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
                m0 = np.random.uniform(0.008, 0.001, numdemes * (numdemes - 1) / 2)
                x0 = np.hstack((N0_inv, m0))
            lims = [(1e-10, 0.01)] * numdemes
            lims += [(0, 0.1)] * (numdemes * (numdemes - 1) / 2)
            try:
                xopt, fun, dic = opt.fmin_l_bfgs_b(compute_Frob_norm_mig, x0, bounds=lims, args=(t[i],
                 obs_rates[:, i], P0, make_merged_pd(pdlist), logVal), approx_grad=True, maxfun=100000, 
                 factr=FTOL, epsilon=EPSILON, iprint=-1)
                if fun < bestfval:
                    bestfval = fun
                    bestxopt = xopt
            except Exception as e:
                print 'Error:', e.message

            N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
            m0 = np.random.uniform(1e-08, 0.001, numdemes * (numdemes - 1) / 2)
            x0 = np.hstack((N0_inv, m0))
            nrestarts = 0
            try:
                xopt, fun, dic = opt.fmin_l_bfgs_b(compute_Frob_norm_mig, x0, bounds=lims, args=(t[i],
                 obs_rates[:, i], P0, make_merged_pd(pdlist), logVal), approx_grad=True, maxfun=100000, 
                 factr=FTOL, epsilon=EPSILON, iprint=-1)
                if fun < bestfval:
                    bestxopt = xopt
                    bestfval = fun
            except Exception as e:
                print 'Error:', e.message

            while nrestarts < RESTARTS and bestfval > FLIMIT:
                N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
                m0 = np.random.uniform(1e-08, 0.001, numdemes * (numdemes - 1) / 2)
                x0 = np.hstack((N0_inv, m0))
                try:
                    xopt, fun, dic = opt.fmin_l_bfgs_b(compute_Frob_norm_mig, x0, bounds=lims, args=(t[i],
                     obs_rates[:, i], P0, make_merged_pd(pdlist), logVal), approx_grad=True, maxfun=100000, 
                     factr=FTOL, epsilon=EPSILON, iprint=-1)
                    if fun < bestfval:
                        bestfval = fun
                        bestxopt = xopt
                except Exception as e:
                    print 'Error:', e.message

                nrestarts += 1

            if verbose:
                print 'Diagnostics:', bestfval, i, bestxopt[0:numdemes], bestxopt[numdemes:]
            Ne_inv = bestxopt[0:numdemes]
            mtemp = bestxopt[numdemes:]
            (state, popdict) = find_pop_merges(Ne_inv, mtemp, t[i], P0, merge_threshold, useMigration, window=window, hack=hack)
            reestimate = False
            if len(popdict) < numdemes and window == 0:
                print 'Merging populations and reestimating parameters:', popdict
                Ne_inv = bestxopt[0:numdemes]
                mtemp = bestxopt[numdemes:]
                m = np.zeros((numdemes, numdemes))
                cnt = 0
                for ii in xrange(numdemes):
                    for jj in xrange(ii + 1, numdemes):
                        m[ii, jj] = m[jj, ii] = mtemp[cnt]
                        cnt += 1

                Qtemp = comp_pw_coal_cont(m, Ne_inv)
                Ptemp = expM(t[i] * Qtemp)
                if verbose:
                    print np.real(P0 * Ptemp)[0:-1, -1]
                    print np.real(obs_rates[:, i])
                print bestxopt
                P0 = converge_pops(popdict, P0)
                reestimate = True
                pdlist.append(popdict)
                numdemes = len(popdict)
                nr = numdemes * (numdemes + 1) / 2 + 1
                lims[:] = []
                bestfval = 1e+200
                bestxopt = None
            elif window > 0 and state != None and len(popdict) < numdemes:
                #merger with the window back approach
                del xopts[-2:] # remove the last 2 estimates
                i = i - 2 # go back 2 timeslices
                P0 = state[0]
                P0 = converge_pops(popdict, P0)
                reestimate = True
                print popdict
                print state[0]
                print i
                pdlist.append(popdict)
                numdemes = len(popdict)
                nr = numdemes*(numdemes+1)/2 + 1
                lims[:] = []
                bestfval=1e+200
                bestxopt = None
            else:
                modXopt = [ 1.0 / x for x in bestxopt[0:numdemes] ]
                cnt = 0
                for ii in xrange(numdemes):
                    for jj in xrange(ii + 1, numdemes):
                        modXopt.append(bestxopt[numdemes + cnt])
                        cnt = cnt + 1
                xopts.append(np.array(modXopt))
                Ne_inv = bestxopt[0:numdemes]
                mtemp = np.abs(bestxopt[numdemes:])

        m = np.zeros((numdemes, numdemes))
        cnt = 0
        for ii in xrange(numdemes):
            for jj in xrange(ii + 1, numdemes):
                if mtemp[cnt] > 0:
                    m[ii, jj] = m[jj, ii] = mtemp[cnt]
                cnt += 1

        Q = comp_pw_coal_cont(m, Ne_inv)
        P = expM(t[i] * Q)
        if verbose:
            print np.real(P0 * P)[0:-1, -1]
            print np.real(obs_rates[:, i])
        P0 = P0 * conv_scrambling_matrix(P)
        i += 1
    if hasattr(find_pop_merges, "states_1"):
        find_pop_merges.states_1 = None
        print 'resetting state 1'
    if hasattr(find_pop_merges, "states_2"):
        find_pop_merges.states_2 = None
        print 'resetting state 2'
    return (xopts, pdlist)

def mp2np(mat):
    """
    Convert from mpmath matrix to np matrix.
    """
    nmat = np.matrix(np.zeros((mat.rows, mat.cols), dtype='float64'))
    for r in xrange(mat.rows):
        for c in xrange(mat.cols):
            nmat[r, c] = mat[r, c]

    return nmat

def popIndices(pops, numPops):
    """La la landis
    """
    indices = np.zeros(len(pops) * (len(pops) + 1) / 2, dtype='int')
    for i in xrange(len(pops)):
        indices[i] = pops[i]

    p = len(pops)
    for i in range(len(pops)):
        for j in range(i + 1, len(pops)):
            indices[p] = pops[i] * (2 * numPops - pops[i] - 3) / 2 + pops[j] - 1 + numPops
            p += 1

    return indices


################################################
# These are helper functions, not needed for   #
# parameter estimation.                        #
################################################
def plot_truth(obs_rates, N1s, N2s, m, t, slicenum = 0):
    """Plot the loss function for the all the parameters.
    """
    nr = np.shape(obs_rates)[0] + 1
    numdemes = int(np.real(np.sqrt(8 * nr - 7)) / 2)
    print 'Starting iterations'
    orates = obs_rates[:, slicenum]
    P0 = np.matrix(np.eye(nr))
    N1s = [ 1.0 / x for x in N1s ]
    N2s = [ 1.0 / x for x in N2s ]
    cnt = 0
    fvals = np.zeros((len(N1s), len(N2s), len(m)))
    for ms in xrange(len(m)):
        for N1 in xrange(len(N1s)):
            for N2 in xrange(len(N2s)):
                xm = np.hstack(([N1s[N1], N2s[N2]], m[ms]))
                fvals[N1, N2, ms] = compute_Frob_norm_mig(xm, t, orates, P0)
                cnt += 1

        if cnt % 1600 == 0:
            print 'Done with ', cnt, 'm values'

    return fvals

def getCoalescentRates(coalProb, gens):
    """This function converts the coalescent probabilities
    to per generation coalescent rates.
    """
    p = np.array(coalProb)
    rates = -np.log(1-p)
    rates = rates/gens
    return rates

def initStartingPoint(obs_rates, nd, t, merged_pd, P0):
    """Choose an initial starting point using a pairwise
    estimate of migration rates and pop sizes.
    """
    x0 = np.zeros(nd * (nd + 1) / 2)
    for i in xrange(nd):
        pass

