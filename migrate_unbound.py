#Embedded file name: Code/migrate_unbound.py
import numpy as np
import pickle
from scipy import linalg
import scipy.optimize as opt
import boundNM as bnm
import sys
from migrate import *

def check_bounds(N, m, obs_coal_rates):
    """Check the bounds on the parameters and return large
    value if not in bounds.
    """
    x = np.hstack((N, m))
    ul = 0.1
    xlim = np.max(np.vstack((np.zeros(np.shape(x)), x - np.ones(np.shape(x)) * ul)), 0)
    if np.max(x) > ul:
        a = linalg.norm(obs_coal_rates, 2) ** 2 + linalg.norm(xlim, 2) ** 2
        return (a, True)
    return (-1, False)


def compute_Frob_norm_mig_unbound(x, t, obs_coal_rates, P0, popdict):
    """Function to compute the order 2 norm distance between the 
    observed coalescent intensities and coalescent intensities 
    computed using the given N and m
    The input is given as x - a vector with N and m values
    P0 is the initial scrambling matrix. Note that we work with 1/N and
    m values, as preconditioning to make the problem more spherical.
    """
    k = int((np.sqrt(1 + 8 * len(x)) - 1) / 2.0)
    Ne_inv = x[0:k] ** 2
    ms = x[k:] ** 2
    f, out_omega = check_bounds(Ne_inv, ms, obs_coal_rates)
    if out_omega:
        return f
    m = np.zeros((k, k))
    cnt = 0
    for i in xrange(k):
        for j in xrange(i + 1, k):
            m[i, j] = m[j, i] = ms[cnt]
            cnt += 1

    Q = comp_pw_coal_cont(m, Ne_inv)
    try:
        Pcurr = np.abs(np.round(expM(t * Q), 12))
    except ValueError as v:
        print 'IN FUNC CFNM'
        raise v

    dims = np.shape(P0)
    Ck = np.matrix(np.eye(dims[0]))[0:dims[0] - 1, :]
    Dk = np.matrix(np.eye(dims[1]))[dims[1] - 1, :]
    Pcurr = Ck * P0 * Pcurr * Dk.T
    if np.shape(obs_coal_rates) != np.shape(Pcurr):
        print 'Dimensions: Observed -> ' + str(np.shape(obs_coal_rates)) + ' and Computed -> ' + str(np.shape(Pcurr))
        raise Exception('Dimensions of observed and computed vectors do not match.')
    try:
        tempo = linalg.norm(average_coal_rates(obs_coal_rates, popdict) - average_coal_rates(Pcurr, popdict), 2) ** 2
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


def grad_Frob_diff(x, curr_value, t, obs_coal_rates, P0, popdict, epsilon = 1e-11, method = 'fdiff'):
    """This function calculates the gradient of the function.
    """
    k = int((np.sqrt(1 + 8 * len(x)) - 1) / 2.0)
    grad = np.zeros(k * (k + 1) / 2)
    dims = np.shape(P0)
    Ck = np.matrix(np.eye(dims[0]))[0:dims[0] - 1, :]
    Dk = np.matrix(np.eye(dims[1]))[dims[1] - 1, :]
    for var in range(len(grad)):
        localx = x.copy()
        localx[var] = localx[var] + epsilon
        Ninv = localx[0:k].copy()
        ms = localx[k:].copy()
        m = np.zeros((k, k))
        cnt = 0
        for i in xrange(k):
            for j in xrange(i + 1, k):
                m[i, j] = m[j, i] = ms[cnt]
                cnt += 1

        Q1 = comp_pw_coal_cont(m, Ninv)
        Pcurr1 = np.abs(np.round(expM(t * Q1), 16))
        Pcurr1 = Ck * P0 * Pcurr1 * Dk.T
        new_value1 = linalg.norm(np.log(average_coal_rates(obs_coal_rates, popdict) + 1e-200) - np.log(average_coal_rates(Pcurr1, popdict) + 1e-200), 2) ** 2
        if method == 'fdiff':
            grad[var] = (new_value1 - curr_value) / epsilon
        elif method == 'cdiff':
            localx[var] = localx[var] - 2 * epsilon
            Ninv = localx[0:k].copy()
            ms = localx[k:].copy()
            m = np.zeros((k, k))
            cnt = 0
            for i in xrange(k):
                for j in xrange(i + 1, k):
                    m[i, j] = m[j, i] = ms[cnt]
                    cnt += 1

            Q2 = comp_pw_coal_cont(m, Ninv)
            Pcurr2 = np.abs(np.round(expM(t * Q2), 16))
            Pcurr2 = Ck * P0 * Pcurr2 * Dk.T
            new_value2 = linalg.norm(np.log(average_coal_rates(obs_coal_rates, popdict) + 1e-200) - np.log(average_coal_rates(Pcurr2, popdict) + 1e-200), 2) ** 2
            grad[var] = (new_value1 - new_value2) / (2 * epsilon)
        else:
            raise Exception('Method "' + method + '" does not exist. Must be "cdiff" or "fdiff".')

    return grad


def comp_N_m_bfgs(obs_rates, t, merge_threshold, useMigration):
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
    nr = np.shape(obs_rates)[0] + 1
    numdemes = int(np.real(np.sqrt(8 * nr - 7)) / 2)
    print 'Starting iterations'
    P0 = np.matrix(np.eye(nr))
    xopts = []
    pdlist = []
    bestxopt = None
    bestfval = 1e+200
    for i in xrange(numslices):
        print 'Running for slice ', i
        if i > 0:
            xopt = bestxopt.copy()
            bestxopt = None
            bestfval = 1e+200
            x0 = xopt.copy()
            xopt, fun, gopt, Bopt, fc, gc, wf = opt.fmin_bfgs(compute_Frob_norm_mig_unbound, x0, args=(t[i],
             obs_rates[:, i],
             P0,
             make_merged_pd(pdlist)), maxiter=100000, gtol=PTOL, epsilon=EPSILON, disp=False, full_output=True)
            bestxopt = xopt
            bestfval = fun
            for lll in xrange(numdemes * (numdemes - 1) / 2):
                x01 = x0.copy()
                x01[numdemes + lll] = 0.0099
                xopt, fun, gopt, Bopt, fc, gc, wf = opt.fmin_bfgs(compute_Frob_norm_mig_unbound, x0, args=(t[i],
                 obs_rates[:, i],
                 P0,
                 make_merged_pd(pdlist)), maxiter=100000, gtol=PTOL, epsilon=EPSILON, disp=False, full_output=True)
                if fun < bestfval:
                    bestfval = fun
                    bestxopt = xopt

        reestimate = True
        while reestimate:
            pdslice = make_merged_pd(pdlist)
            print 'pdslice:', pdslice
            N0_inv = np.random.uniform(0.05, 0.1, numdemes)
            m0 = np.random.uniform(0.0001, 0.03, numdemes * (numdemes - 1) / 2)
            x0 = np.hstack((N0_inv, m0))
            xopt, fun, gopt, Bopt, fc, gc, wf = opt.fmin_bfgs(compute_Frob_norm_mig_unbound, x0, args=(t[i],
             obs_rates[:, i],
             P0,
             make_merged_pd(pdlist)), maxiter=100000, gtol=PTOL, epsilon=EPSILON, disp=False, full_output=True)
            if fun < bestfval:
                bestfval = fun
                bestxopt = xopt
                print gopt
            N0_inv = np.random.uniform(0.05, 0.1, numdemes)
            m0 = np.random.uniform(0.0001, 0.03, numdemes * (numdemes - 1) / 2)
            x0 = np.hstack((N0_inv, m0))
            nrestarts = 0
            xopt, fun, gopt, Bopt, fc, gc, wf = opt.fmin_bfgs(compute_Frob_norm_mig_unbound, x0, args=(t[i],
             obs_rates[:, i],
             P0,
             make_merged_pd(pdlist)), maxiter=100000, gtol=PTOL, epsilon=EPSILON, disp=False, full_output=True)
            if fun < bestfval:
                bestxopt = xopt
                bestfval = fun
                print gopt
            print nrestarts
            while nrestarts < RESTARTS and bestfval > FLIMIT:
                N0_inv = np.random.uniform(5e-05, 0.001, numdemes)
                m0 = np.random.uniform(1e-08, 0.001, numdemes * (numdemes - 1) / 2)
                x0 = np.hstack((N0_inv, m0))
                xopt, fun, gopt, Bopt, fc, gc, wf = opt.fmin_bfgs(compute_Frob_norm_mig_unbound, x0, args=(t[i],
                 obs_rates[:, i],
                 P0,
                 make_merged_pd(pdlist)), maxiter=100000, gtol=PTOL, epsilon=EPSILON, disp=False, full_output=True)
                if fun < bestfval:
                    bestfval = fun
                    bestxopt = xopt
                    print gopt
                print nrestarts
                nrestarts += 1

            print bestfval, i, bestxopt[0:numdemes] ** 2, bestxopt[numdemes:] ** 2
            Ne_inv = bestxopt[0:numdemes]
            mtemp = bestxopt[numdemes:]
            popdict = find_pop_merges(Ne_inv ** 2, mtemp ** 2, t[i], P0, merge_threshold, useMigration)
            reestimate = False
            if len(popdict) < numdemes:
                print 'Merging populations and reestimating parameters:', popdict
                print bestxopt
                P0 = converge_pops(popdict, P0)
                reestimate = True
                pdlist.append(popdict)
                numdemes = len(popdict)
                nr = numdemes * (numdemes + 1) / 2 + 1
                bestfval = 1e+200
                bestxopt = None
            else:
                modXopt = [ 1.0 / x ** 2 for x in bestxopt[0:numdemes] ]
                cnt = 0
                for ii in xrange(numdemes):
                    for jj in xrange(ii + 1, numdemes):
                        modXopt.append(bestxopt[numdemes + cnt] ** 2)
                        cnt = cnt + 1

                xopts.append(np.array(modXopt))
                Ne_inv = bestxopt[0:numdemes] ** 2
                mtemp = bestxopt[numdemes:] ** 2

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