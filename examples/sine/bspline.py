#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np


def evalBsplines(degree, deltaKnots, time, outvec):

    # Get interval index l s.t. t \in [t_l, t_l+1]
    l = int(time / deltaKnots)   # this will round down to next smaller integer

    # Set outvec to unit vector 1, 0, 0, 0...
    outvec[:] = [0.0] * len(outvec)
    outvec[0] = 1.0

    # Recursive loop to update splines
    for i in range(1,degree+1):        # i = 1,2,...,degree
        for r in range(i,0,-1):        # r = i, i-1, ..., 1
            coeff1 = (time - (l-i+r)*deltaKnots)  / ( (l+r)*deltaKnots - (l-i+r)*deltaKnots )
            coeff2 = ( (l+r+1)*deltaKnots - time) / ( (l+r+1)*deltaKnots - (l-i+r+1)*deltaKnots )
            outvec[r] = coeff1 * outvec[r-1] + coeff2 * outvec[r]
        outvec[0] = outvec[0] * ((l+1)*deltaKnots - time) / ((l+1)*deltaKnots - (l-i+1)*deltaKnots)


def spline_test(degree, nKnots, Tfinal, deltax):
    print("Testing Bsplines(degree=", degree, "), nKnots=", nKnots)

    # Init grid and splines
    n = int(Tfinal / deltax)
    xgrid = np.linspace(0.0, Tfinal, n+1)
    nsplines = nKnots + degree
    spline = np.zeros((nsplines)*(n+1)).reshape(nsplines, n+1)
    deltaKnots = Tfinal / (nKnots-1)

    # Loop over time domain and compute spline coefficients
    for i in range(len(xgrid)):

        time = xgrid[i]
        l = int(time / deltaKnots)
        # print(spline[l:l+degree+1,i].shape)
        evalBsplines(degree, deltaKnots, time, spline[l:l+degree+1, i])

    # Plot
    for i in range(nsplines-1):
        plt.plot(xgrid, spline[i,:], label="B_"+str(i))
        # print(i, " ", spline[i,:])
    plt.legend()
    plt.show()


# Test now:
# degree = 2
# nKnots = 5
# Tfinal = 1.0
# deltax = 0.01
# spline_test(degree, nKnots, Tfinal, deltax)
