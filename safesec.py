#Embedded file name: Code/safesec.py
"""
file                       safesec.py
version    0.0.1 mar 23, 2009 initial release
author          Ernesto P. Adorio, Ph.D.
                                   UPDEPP at Clark Field, Pampanga
description        Given a function f,
                                  returns a root or zero of the funciton such that f(x)= 0.
                                  by using a combination of the secant and half-interval methods.
references              Numerical Recipes by Press, et. al.
"""
from math import *
from scipy.linalg import norm

def issamesign(a, b):
    if a < 0.0 and b < 0.0 or a > 0.0 and b > 0.0:
        return True
    return False


def findsignchange(f, xl, xr, mindx = 0.01, maxfeval = 1000, verbose = False):
    """
        Arguments:
                f         -   function of x
                xl, xr -   initial bounds
                mindx  -   minimum interval width
                maxfeval - maximum number of iterations allowed.
                verbose  -   shall debbuging statements be printed?
        Return value:
           (xl, xr,neval) interval where there is a change in sign of the function
                                          neval > maxfeval is an indicator of FAILURE.
           None  Failure
    """
    print 'xl, xr = ', xl, xr
    fxl = f(xl)
    fxr = f(xr)
    dx = xr - xl
    dxhalf = dx * 0.5
    neval = 0
    if not issamesign(fxl, fxr):
        return (xl, xr, 0)
    Ok = True
    while Ok:
        x = xl + dxhalf
        while True:
            if x > xr:
                break
            fx = f(x)
            neval += 1
            if verbose:
                print 'neval, x, f(x)=', neval, x, f(x)
            if neval > maxfeval:
                Ok = False
                break
            if not issamesign(fxl, fx):
                xr = x
                xl = x - dxhalf
                break
            x += dx

        dx = dxhalf
        dxhalf = dxhalf * 0.5