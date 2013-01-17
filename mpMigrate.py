#Embedded file name: Code/mpMigrate.py
"""
This file implements the parameter esimtation for the 
migration project using the mpmath library.
"""
import numpy as np
import pickle
from scipy import linalg
import scipy.optimize as opt
import boundNM as bnm
import sys
import mpmath as mp
mp.dps = 24
mp.pretty = True

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
        raise Exception('Migration matrix and population size vector indicate different number of demes.')
    nr = numdemes * (numdemes + 1) / 2 + 1
    Q = mp.zeros(nr, nr)
    rnum = -1
    demePairs = {}
    for p1 in xrange(numdemes):
        for p2 in xrange(p1, numdemes):
            rnum += 1
            demePairs[p1, p2] = rnum

    for dp1 in demePairs.keys():
        if dp1[0] == dp1[1]:
            Q[demePairs[dp1], nr - 1] = 0.5 * Ne_inv[dp1[0]]
        for dp2 in demePairs.keys():
            if dp1 == dp2:
                continue
            if dp1[0] == dp1[1]:
                if dp2[0] == dp2[1]:
                    continue
                elif dp2[0] == dp1[0]:
                    Q[demePairs[dp1], demePairs[dp2]] = 2 * m[dp1[1]][dp2[1]]
                elif dp2[1] == dp1[1]:
                    Q[demePairs[dp1], demePairs[dp2]] = 2 * m[dp1[0]][dp2[0]]
            elif dp2[0] in dp1:
                if dp1[0] == dp2[0]:
                    Q[demePairs[dp1], demePairs[dp2]] = m[dp1[1]][dp2[1]]
                else:
                    Q[demePairs[dp1], demePairs[dp2]] = m[dp1[0]][dp2[1]]
            elif dp2[1] in dp1:
                if dp1[0] == dp2[1]:
                    Q[demePairs[dp1], demePairs[dp2]] = m[dp1[1]][dp2[0]]
                else:
                    Q[demePairs[dp1], demePairs[dp2]] = m[dp1[0]][dp2[0]]

    for dp1 in demePairs.keys():
        Q[demePairs[dp1], demePairs[dp1]] = -sum(Q[demePairs[dp1], :])

    return Q


def compute_Frob_norm_mig(x, t, obs_coal_rates, P0, popdict):
    """Function to compute the order 2 norm distance between the 
    observed coalescent intensities and coalescent intensities 
    computed using the given N and m
    The input is given as x - a vector with N and m values
    P0 is the initial scrambling matrix. Note that we work with 1/N and
    m values, as preconditioning to make the problem more spherical.
    """
    print x
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
        Pcurr = expM(t * Q)
    except ValueError as v:
        print 'OUT FUNC CFNM, trouble baby'
        raise v

    dims = np.shape(P0)
    Ck = np.matrix(np.eye(dims[0]))[0:dims[0] - 1, :]
    Dk = np.matrix(np.eye(dims[1]))[dims[1] - 1, :]
    Pcurr = Ck * P0 * Pcurr * Dk.T
    if np.shape(obs_coal_rates) != np.shape(Pcurr):
        print 'Dimensions: Observed -> ' + str(np.shape(obs_coal_rates)) + ' and Computed -> ' + str(np.shape(Pcurr))
        raise Exception('Dimensions of observed and computed vectors do not match.')
    try:
        tempo = linalg.norm(np.log(average_coal_rates(obs_coal_rates, popdict) + 1e-200) - np.log(average_coal_rates(Pcurr, popdict) + 1e-200), 2) ** 2
    except:
        print 'obsRates', obs_coal_rates
        print 'average', average_coal_rates(obs_coal_rates, popdict)
        print 'curr', Pcurr
        print 'Q', Q
        print 'P', expM(t * Q)
        print 'P0', P0
        print 'Error in tempo'
        raise 

    return tempo


def conv_scrambling_matrix(P):
    """This takes a Probability matrix and converts it to a scrambling matrix,
    to get the probability of lines moving from the current demes to any config
    of demes at the beginning of the time slice.
    """
    Pnew = P.copy()
    rw = Pnew.rows
    cl = Pnew.cols
    Pnew[0:rw - 1, cl - 1] = 0
    for r in xrange(rw - 1):
        rsum = sum(Pnew[r, :])
        Pnew[r, :] = Pnew[r, :] / rsum

    return Pnew


def converge_pops(popDict, scramb):
    """Changes the scrambling matrix appropriately upon
    the merging of populations.
    """
    npop = scramb.cols
    npop = int(np.real(np.sqrt(8 * npop - 7)) / 2)
    nr = scramb.rows
    nc = len(popDict)
    nc = nc * (nc + 1) / 2 + 1
    temp = mp.zeros(nr, nc)
    temp[:, -1] = scramb[:, -1]
    ptc = pop_to_col(popDict, npop)
    for j in xrange(nc - 1):
        for k in xrange(len(ptc[j])):
            temp[:, j] = temp[:, j] + scramb[:, ptc[j][k]]

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
                    if a > b:
                        continue
                    tt = a * (2 * nd_old - a + 1) / 2 + b - a
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
    exp_rates = mp.zeros(nr - 1, numslices)
    P0 = mp.eye(nr)
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
            raise Exception('Population map and other parameters do not match ' + str(i))
        for ii in xrange(numpops):
            for jj in xrange(ii + 1, numpops):
                m[ii, jj] = m[jj, ii] = mtemp[cnt]
                cnt += 1

        Q = comp_pw_coal_cont(m, Ne_inv)
        eQ = expM(ts[i] * Q)
        P = P0 * eQ
        exp_rates[:, i] = P[0:nr - 1, P.cols - 1]
        P0 = P0 * conv_scrambling_matrix(eQ)
        for r in xrange(exp_rates.rows):
            for c in xrange(exp_rates.cols):
                if exp_rates[r, c] < 0:
                    exp_rates[r, c] = 0

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

    popVals = np.unique(popToIndex.values())
    popmap = []
    for ii in xrange(currIndex):
        temp = []
        for jj in popToIndex:
            if popToIndex[jj] == ii:
                temp.append(jj)

        popmap.append(tuple(temp))

    return popmap


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


def find_pop_merges(Ninv, mtemp, t, P0, merge_threshold, useMigration):
    """This function takes the optimal paramters found and 
    figures out if the populations need to be merged.
    """
    numdemes = len(Ninv)
    if useMigration == False:
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
                meanRates = np.mean(dists)
                medianRates = np.median(dists)
                rangeRates = np.max(dists) - np.min(dists)
                print i, j, medianRates, meanRates
                if rangeRates / meanRates < merge_threshold:
                    print 'Hello:', i, j, np.real(dists)
                    popdict[i].append(j)
                    popdict[j].append(i)

    else:
        print 'Using migration rates'
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

    return construct_poparr(popdict)


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


def comp_N_m_mp(obs_rates, t, merge_threshold, useMigration):
    """This function estimates the N and m parameters for the various time slices. The 
    time slices are given in a vector form 't'. t specifies the length of the time slice, not
    the time from present to end of time slice (not cumulative but atomic)
    Also, the obs_rates are given, for each time slice are given in columns of the obs_rates
    matrix. Both obs_rates and time slice lengths are given from present to past.
    """
    FTOL = 10.0
    PTOL = 1e-20
    EPSILON = 1e-11
    RESTARTS = 10
    FLIMIT = 1e-20
    numslices = len(t)
    nr = obs_rates.rows + 1
    numdemes = int(np.real(np.sqrt(8 * nr - 7)) / 2)
    print 'Starting iterations'
    P0 = mp.eye(nr)
    xopts = []
    pdlist = []
    for i in xrange(numslices):
        bestxopt = None
        bestfval = 1e+200
        print 'Running for slice ', i
        if i > 0:
            x0 = xopt[0:nr - 1].copy()
            try:
                xopt = mp.findroot(lambda x: compute_Frob_norm_mig(x, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist)), x0)
                fun = compute_Frob_norm_mig(xopt, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist))
                bestxopt = xopt
                bestfval = fun
            except ValueError as e:
                print 'Error:', e.message

            for lll in xrange(numdemes * (numdemes - 1) / 2):
                x01 = x0[0:nr - 1].copy()
                x01[numdemes + lll] = 0.0099
                try:
                    xopt = mp.findroot(lambda x: compute_Frob_norm_mig(x, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist)), x01)
                    fun = compute_Frob_norm_mig(xopt, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist))
                    if fun < bestfval:
                        bestfval = fun
                        bestxopt = xopt
                except ValueError as e:
                    print 'Error:', e.message

        reestimate = True
        while reestimate:
            pdslice = make_merged_pd(pdlist)
            print 'pdslice:', pdslice
            N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
            m0 = np.random.uniform(0.008, 0.001, numdemes * (numdemes - 1) / 2)
            x0 = [ mp.convert(rrr) for rrr in N0_inv ]
            for zz in range(len(m0)):
                x0.append(mp.convert(m0[zz]))

            lims = [(1e-15, 0.1)] * numdemes
            lims += [(1e-15, 0.1)] * (numdemes * (numdemes - 1) / 2)
            try:
                xopt = mp.findroot(lambda x: compute_Frob_norm_mig(x, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist)), x0)
                fun = compute_Frob_norm_mig(xopt, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist))
                if fun < bestfval:
                    bestfval = fun
                    bestxopt = xopt
            except ValueError as e:
                print 'Error:', e.message

            N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
            m0 = np.random.uniform(1e-08, 0.001, numdemes * (numdemes - 1) / 2)
            x0 = [ mp.convert(rrr) for rrr in N0_inv ]
            for zz in range(len(m0)):
                x0.append(mp.convert(m0[zz]))

            nrestarts = 0
            try:
                xopt = mp.findroot(lambda x: compute_Frob_norm_mig(x, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist)), x0)
                fun = compute_Frob_norm_mig(xopt, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist))
                if fun < bestfval:
                    bestxopt = xopt
                    bestfval = fun
            except ValueError as e:
                print 'Error:', e.message

            print nrestarts
            while nrestarts < RESTARTS and bestfval > FLIMIT:
                N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
                m0 = np.random.uniform(1e-08, 0.001, numdemes * (numdemes - 1) / 2)
                x0 = mp.zeros(len(N0_inv) + len(m0))
                for zz in range(len(N0_inv)):
                    x0[zz] = N0_inv[zz]

                for zz in range(len(m0)):
                    x0[zz + len(N0_inv)] = m0[i]

                try:
                    xopt = mp.findroot(lambda x: compute_Frob_norm_mig(x, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist)), x0)
                    fun = compute_Frob_norm_mig(xopt, t[i], obs_rates[:, i], P0, make_merged_pd(pdlist))
                    if fun < bestfval:
                        bestfval = fun
                        bestxopt = xopt
                except ValueError as e:
                    print 'Error:', e.message

                print nrestarts
                nrestarts += 1

            print bestfval, i, bestxopt[0:numdemes], bestxopt[numdemes:]
            Ne_inv = bestxopt[0:numdemes]
            mtemp = bestxopt[numdemes:]
            popdict = find_pop_merges(Ne_inv, mtemp, t[i], P0, merge_threshold, useMigration)
            reestimate = False
            if len(popdict) < numdemes:
                print 'Merging populations and reestimating parameters:', popdict
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
                modXopt = [ 1.0 / x for x in bestxopt[0:numdemes] ]
                cnt = 0
                for ii in xrange(numdemes):
                    for jj in xrange(ii + 1, numdemes):
                        modXopt.append(bestxopt[numdemes + cnt])
                        cnt = cnt + 1

                xopts.append(np.array(modXopt))
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
        print 'est rates', i
        print np.real(P0 * P)[0:-1, -1]
        print 'obs rates', i
        print np.real(obs_rates[:, i])
        print 'Min func value', bestfval
        P0 = P0 * conv_scrambling_matrix(P)
        ist = raw_input('Waiting for input...')

    return (xopts, pdlist)


def expM(Q):
    """Uses mpmath's exp computation
    """
    return mp.expm(Q)