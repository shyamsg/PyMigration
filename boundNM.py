#Embedded file name: /mnt/lustre/home/shyamg/projects/Migration/Code/boundNM.py
import numpy
from numpy import atleast_1d, eye, mgrid, argmin, zeros, shape, squeeze, vectorize, asarray, absolute, sqrt, Inf, asfarray, isinf
_status_message = {'success': 'Optimization terminated successfully.',
 'maxfev': 'Maximum number of function evaluations has been exceeded.',
 'maxiter': 'Maximum number of iterations has been exceeded.',
 'pr_loss': 'Desired error not necessarily achieved due to precision loss.'}

def fmin_bound(func, x0, args = (), bounds = None, xtol = 0.0001, ftol = 0.0001, maxiter = None, maxfun = None, full_output = 0, disp = 1, retall = 0, callback = None):
    """
    Minimize a function using the downhill simplex algorithm.
    
    This algorithm only uses function values, not derivatives or second
    derivatives.
    
    Parameters
    ----------
    func : callable func(x,*args)
    The objective function to be minimized.
    x0 : ndarray
    Initial guess.
    args : tuple
    Extra arguments passed to func, i.e. ``f(x,*args)``.
    callback : callable
    Called after each iteration, as callback(xk), where xk is the
    current parameter vector.
    
    Returns
    -------
    xopt : ndarray
    Parameter that minimizes function.
    fopt : float
    Value of function at minimum: ``fopt = func(xopt)``.
    iter : int
    Number of iterations performed.
    funcalls : int
    Number of function calls made.
    warnflag : int
    1 : Maximum number of function evaluations made.
    2 : Maximum number of iterations reached.
    allvecs : list
    Solution at each iteration.
    
    Other parameters
    ----------------
    xtol : float
    Relative error in xopt acceptable for convergence.
    ftol : number
    Relative error in func(xopt) acceptable for convergence.
    maxiter : int
    Maximum number of iterations to perform.
    maxfun : number
    Maximum number of function evaluations to make.
    full_output : bool
    Set to True if fopt and warnflag outputs are desired.
    disp : bool
    Set to True to print convergence messages.
    retall : bool
    Set to True to return list of solutions at each iteration.
    
    See also
    --------
    minimize: Interface to minimization algorithms for multivariate
    functions. See the 'Nelder-Mead' `method` in particular.
    
    Notes
    -----
    Uses a Nelder-Mead simplex algorithm to find the minimum of function of
    one or more variables.
    
    This algorithm has a long history of successful use in applications.
    But it will usually be slower than an algorithm that uses first or
    second derivative information. In practice it can have poor
    performance in high-dimensional problems and is not robust to
    minimizing complicated functions. Additionally, there currently is no
    complete theory describing when the algorithm will successfully
    converge to the minimum, or how fast it will if it does.
    
    References
    ----------
    Nelder, J.A. and Mead, R. (1965), "A simplex method for function
    minimization", The Computer Journal, 7, pp. 308-313
    Wright, M.H. (1996), "Direct Search Methods: Once Scorned, Now
    Respectable", in Numerical Analysis 1995, Proceedings of the
    1995 Dundee Biennial Conference in Numerical Analysis, D.F.
    Griffiths and G.A. Watson (Eds.), Addison Wesley Longman,
    Harlow, UK, pp. 191-208.
    
    """
    opts = {'xtol': xtol,
     'ftol': ftol,
     'maxiter': maxiter,
     'maxfev': maxfun,
     'disp': disp}
    if retall and not full_output:
        out = _minimize_neldermead(func, x0, args, bounds, opts, full_output=True, retall=retall, callback=callback)
    else:
        out = _minimize_neldermead(func, x0, args, bounds, opts, full_output, retall, callback)
    if full_output:
        x, info = out
        retlist = (x,
         info['fun'],
         info['nit'],
         info['nfev'],
         info['status'])
        if retall:
            retlist += (info['allvecs'],)
        return retlist
    elif retall:
        x, info = out
        return (x, info['allvecs'])
    else:
        return out


def _minimize_neldermead(func, x0, args = (), bounds = None, options = {}, full_output = 0, retall = 0, callback = None):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.
    
    Options for the Nelder-Mead algorithm are:
    disp : bool
    Set to True to print convergence messages.
    xtol : float
    Relative error in solution `xopt` acceptable for convergence.
    ftol : float
    Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter : int
    Maximum number of iterations to perform.
    maxfev : int
    Maximum number of function evaluations to make.
    
    This function is called by the `minimize` function with
    `method=Nelder-Mead`. It is not supposed to be called directly.
    """
    xtol = options.get('xtol', 0.0001)
    ftol = options.get('ftol', 0.0001)
    maxiter = options.get('maxiter')
    maxfun = options.get('maxfev')
    disp = options.get('disp', False)
    fcalls, func = wrap_function(func, args)
    x0 = asfarray(x0).flatten()
    x0 = check_bounds(x0, bounds)
    N = len(x0)
    rank = len(x0.shape)
    if not -1 < rank < 2:
        raise ValueError('Initial guess must be a scalar or rank-1 sequence.')
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200
    if bounds != None and len(bounds) != N:
        raise AssertionError('Bounds set incorrectly.')
    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    one2np1 = range(1, N + 1)
    if rank == 0:
        sim = numpy.zeros((N + 1,), dtype=x0.dtype)
    else:
        sim = numpy.zeros((N + 1, N), dtype=x0.dtype)
    fsim = numpy.zeros((N + 1,), float)
    sim[0] = x0
    if retall:
        allvecs = [sim[0]]
    fsim[0] = func(x0)
    nonzdelt = 0.05
    zdelt = 0.00025
    for k in range(0, N):
        y = numpy.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt) * y[k]
        else:
            y[k] = zdelt
        sim[k + 1] = y
        f = func(y)
        fsim[k + 1] = f

    ind = numpy.argsort(fsim)
    fsim = numpy.take(fsim, ind, 0)
    sim = numpy.take(sim, ind, 0)
    iterations = 1
    while fcalls[0] < maxfun and iterations < maxiter:
        if max(numpy.ravel(abs(sim[1:] - sim[0]))) <= xtol and max(abs(fsim[0] - fsim[1:])) <= ftol:
            break
        xbar = numpy.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        xr = check_bounds(xr, bounds)
        fxr = func(xr)
        doshrink = 0
        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            xe = check_bounds(xe, bounds)
            fxe = func(xe)
            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        elif fxr < fsim[-2]:
            sim[-1] = xr
            fsim[-1] = fxr
        else:
            if fxr < fsim[-1]:
                xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                xc = check_bounds(xc, bounds)
                fxc = func(xc)
                if fxc <= fxr:
                    sim[-1] = xc
                    fsim[-1] = fxc
                else:
                    doshrink = 1
            else:
                xcc = (1 - psi) * xbar + psi * sim[-1]
                xcc = check_bounds(xcc, bounds)
                fxcc = func(xcc)
                if fxcc < fsim[-1]:
                    sim[-1] = xcc
                    fsim[-1] = fxcc
                else:
                    doshrink = 1
            if doshrink:
                for j in one2np1:
                    sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                    fsim[j] = func(sim[j])

        ind = numpy.argsort(fsim)
        sim = numpy.take(sim, ind, 0)
        fsim = numpy.take(fsim, ind, 0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = min(fsim)
    warnflag = 0
    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print 'Warning: ' + msg
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print 'Warning: ' + msg
    else:
        msg = _status_message['success']
        if disp:
            print msg
            print ' Current function value: %f' % fval
            print ' Iterations: %d' % iterations
            print ' Function evaluations: %d' % fcalls[0]
    if full_output:
        info = {'fun': fval,
         'nit': iterations,
         'nfev': fcalls[0],
         'status': warnflag,
         'success': warnflag == 0,
         'message': msg,
         'solution': x}
        if retall:
            info['allvecs'] = allvecs
        return (x, info)
    else:
        return x


def check_bounds(x, bounds):
    """This routine checks the bounds on the simplex vertex 
    and adjusts it accordingly by moving the out of bound
    parameter to the boundary.
    """
    myx = x.copy()
    if bounds == None:
        return myx
    for i in xrange(len(myx)):
        if bounds[i] != None:
            if bounds[i][0] != None and bounds[i][1] != None:
                if bounds[i][0] >= bounds[i][1]:
                    raise Exception('Lower bound (' + str(bounds[i][0]) + ') >= Upper bound (' + str(bounds[i][1]) + ').')
            if myx[i] < bounds[i][0] and bounds[i][0] != None:
                myx[i] = bounds[i][0]
            elif myx[i] > bounds[i][1] and bounds[i][1] != None:
                myx[i] = bounds[i][1]

    return myx


def wrap_function(function, args):
    ncalls = [0]

    def function_wrapper(x):
        ncalls[0] += 1
        return function(x, *args)

    return (ncalls, function_wrapper)