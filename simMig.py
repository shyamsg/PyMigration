#Embedded file name: /mnt/lustre/home/shyamg/projects/Migration/Code/simMig.py
"""This script gives function to simulate data given pop sizes, 
migration rates, population history and time slice lengths.
"""
import migrate as mig
import numpy as np
from scipy import linalg
import sys
import os
import re

def add_white_noise(rates, numreg):
    """Given the rates, add noise based on numreg
    """
    rtemp = rates.copy().getA()
    sdrates = np.sqrt(rtemp * (1 - rtemp) / numreg) + 1e-10
    noise = np.random.normal(0, sdrates)
    rtemp += noise
    return np.matrix(rtemp)


def add_uniform_noise(rates, percent):
    """Given the rates, sample new rate uniformly between
    ((1-percent)*rates, (1+percent)*rates)
    """
    raise 0 < percent < 1 or AssertionError
    rtemp = rates.copy().getA()
    noise = np.random.uniform(1 - percent, 1 + percent, np.shape(rtemp))
    rtemp = rtemp * noise
    return np.matrix(rtemp)


def run_Over_Grid(numdemes = 2, reps = 10, numreg = 100, t = 1000):
    """This function runs the estimation procedure for the first time slice for 
    given number of demes and repeats the process reps number of times. The 
    values of mean pop size and mig rates is preset but will be changed in future
    versions. The third parameter here controls the noise amount in the estimates
    of coalescent intensities - number of regions that contributed to the
    estimate itself
    """
    Nmean = 2000
    Nsd = 100
    migMean = 0.0001
    migsd = 1e-06
    ndc2 = numdemes * (numdemes - 1) / 2
    rows = ndc2 + numdemes + 1
    I = np.matrix(np.eye(rows))
    Ck = I[0:rows - 1, :]
    Dk = I[rows - 1, :]
    output = []
    for r in xrange(reps):
        N = np.random.normal(Nmean, Nsd, (numdemes,))
        mtemp = np.random.normal(migMean, migsd, (ndc2,))
        xtrue = np.hstack((N, mtemp))
        m = np.zeros((numdemes, numdemes))
        cnt = 0
        for i in xrange(numdemes):
            for j in xrange(i + 1, numdemes):
                m[i, j] = m[j, i] = mtemp[cnt]
                cnt += 1

        Ninv = [ 1.0 / x for x in N ]
        Qtrue = comp_pw_coal_cont(m, Ninv)
        Ptrue = expM(t * Qtrue)
        obs_rates = Ck * Ptrue * Dk.T
        if numreg > 0:
            sd_rates = np.real(np.sqrt(obs_rates.getA() * (1 - obs_rates).getA() / numreg))
            noise = np.random.normal(0.0, sd_rates)
            print 'Noise:\n', noise
        N0 = np.random.normal(Nmean / 2.0, Nsd * 3.0, (numdemes,))
        m0 = np.random.normal(migMean / 2.0, migsd * 3.0, (ndc2,))
        x0 = np.hstack((N0, m0))
        xopt = opt.fmin(compute_Frob_norm_mig, x0, (t, obs_rates), maxfun=1000000, maxiter=100000)
        output.append((xtrue, xopt, linalg.norm(xopt - xtrue)))

    return output


def run_for_parms(Ns, ms, ts, popmaps, numreg, reps, compError = False, coal_error_threshold = 0.0001):
    """This function runs the estimation procedure given the 
    population sizes, mig rates, times, pop history. numreg 
    controls the noise in the estimate of coal rates,
    and reps repeats the procedure multiple times.
    """
    true_parms = []
    for i in xrange(len(Ns)):
        if len(Ns[i]) > 1:
            true_parms.append(np.array(Ns[i] + ms[i]))
        else:
            true_parms.append(np.array(Ns[i]))

    true_rates = mig.compute_pw_coal_rates(ms, Ns, ts, popmaps)
    print 'True_rates:', true_rates
    xopts = []
    estErr = []
    while reps > 0:
        obs_rates = add_uniform_noise(true_rates, numreg)
        while np.min(obs_rates) < 0 or np.max(obs_rates) > 1:
            obs_rates = add_uniform_noise(true_rates, numreg)

        xopt = mig.comp_N_m(obs_rates, ts, coal_error_threshold)
        xopts.append(xopt)
        reps -= 1
        if compError:
            estErr.append(compute_error(true_parms, xopt))

    if compError:
        return (xopts, estErr)
    else:
        return xopts


def compute_error(true, estimate, order = np.inf):
    """Given the true and the estimated parameter values
    this function computes the error in the parameter 
    estimates. The order controls the norm used, by default
    its the maximum - so sup norm 
    """
    print true
    print estimate
    errs = []
    for i in xrange(len(true)):
        estError = abs(true[i] - estimate[i])
        for j in xrange(len(true[i])):
            if true[i][j] != 0:
                estError[j] = estError[j] / true[i][j]

        errs.append(linalg.norm(estError, order))

    return errs


def process_time_string(timestr):
    """This function processes the timestring from PSMC
    and converts this to list of time slice lengths
    """
    timestr = timestr.strip()
    toks = timestr.split('+')
    timeslices = []
    for t in toks:
        tm = t.strip()
        mobj = re.search('\\*', tm)
        if mobj == None:
            timeslices += [int(tm)]
        else:
            tms = tm.split('*')
            timeslices += int(tms[0]) * [int(tms[1])]

    return timeslices


def mkCoalMatrix(C, npop):
    """The coalescence matrix C as a vectorization of
    the upper triangular matrix and npop, the number of 
    demes.
    """
    C = np.array(C).flatten()
    M = np.zeros((npop, npop))
    cnt = 0
    for i in range(npop):
        for j in range(i, npop):
            M[i, j] = C[cnt]
            if i != j:
                M[j, i] = M[i, j]
            cnt += 1

    return M


class run_single_sim:
    """This class takes the outputs from post processed PSMC,
    and runs our method on it to estimate. Note here that the
    popScaling paramtere = 2N0 and NOT N0. Also the last row
    of rates is dropped.
    """

    def __init__(self, popScaling, ratefile, timeStr, ignoreLast = False, logVal = True, verbose = False):
        """Initialization function of the class.
        """
        self.verbose = verbose
        self.estimatedParms = None
        self.modified = False
        self.obsRates = []
        self.logVal = logVal
        times = []
        popScaling = float(popScaling)
        r = open(ratefile)
        for line in r:
            toks = line.strip().split()
            times.append(float(toks[0]) * popScaling)
            currRates = [ float(x) for x in toks[1:] ]
            self.obsRates.append(currRates)

        self.timeslices = times
        self.timeStr = timeStr
        if ignoreLast:
            self.obsRates = self.obsRates[0:-1]
            self.timeslices = self.timeslices[0:-1]
        self.obsRates = np.matrix(self.obsRates).T

    def modify_rates(self):
        """The rates obtained from PSMC are the prob of coal 
        in that timeslice, not the prob of coal in that timeslice
        AND not coalescing in any other timeslice. We need the 
        conditional probability of coal in that timeslice given
        lines have not coalesced in any of the previous timeslices.
        This function converts the PSMC values into our values.
        """
        if self.modified:
            print 'Already Modified Probabilities'
        else:
            testrates = self.obsRates.copy()
            tratesum = testrates.cumsum(1)
            nocoal = 1 - tratesum
            nocoal = nocoal[:, :-1]
            nocoal = np.hstack((np.ones((np.shape(nocoal)[0], 1)), nocoal))
            testrates = testrates.getA() / (nocoal.getA() + 1e-200)
            self.modified = True
            self.obsRates = np.matrix(np.max([np.min([testrates, np.ones(np.shape(testrates))], 0), np.zeros(np.shape(testrates))], 0))

    def collapse_using_timeStr(self):
        """This function collapses the time slices and 
        the coalescent prbabilities using the time string
        """
        if self.modified == True:
            raise Exception('Probabilities already modified.\nCollapsing after modification will lead to incorrect results.')
        timeUnits = np.array(process_time_string(self.timeStr))
        if len(self.timeslices) + 1 == np.sum(timeUnits):
            if timeUnits[-1] == 1:
                timeUnits = timeUnits[:-1]
            else:
                timeUnits[-1] -= 1
        if len(self.timeslices) != np.sum(timeUnits):
            raise Exception('Total number of timeslices is different.')
        ind = 0
        cnt = 0
        curr_rates = np.matrix(np.zeros((np.shape(self.obsRates)[0], len(timeUnits))))
        curr_times = []
        for i in timeUnits:
            curr_rates[:, cnt] = np.sum(self.obsRates[:, ind:ind + i], axis=1)
            curr_times.append(np.sum(self.timeslices[ind:ind + i]))
            ind += i
            cnt += 1

        self.obsRates = curr_rates
        self.timeslices = curr_times

    def estimate_sim_run(self, merge_threshold = 0.01, useMigration = False, DFO = False, window = 0, hack = False):
        """This function estimates the pop and mig in each
        timeslice and returns it. If useMigration, the threshold
        is the migration threshold, if not the threshold is the
        coal rate threshold
        """
        if DFO:
            self.estimatedParms = mig.comp_N_m(self.obsRates, self.timeslices, merge_threshold, useMigration, self.logVal, self.verbose)
        else:
            self.estimatedParms = mig.comp_N_m_bfgs(self.obsRates, self.timeslices, merge_threshold, useMigration, False, self.logVal, self.verbose, window, hack)
        return self.estimatedParms


if __name__ == '__main__':
    simObj = run_single_sim(sys.argv[1], sys.argv[2])
