#!/usr/bin/env python

class BsplineBasis():
    def __init__(self, nsplines, degree, Tfinal):
        self.nsplines = nsplines
        self.degree = degree
        self.Tf = Tfinal

        self.nKnots = self.nsplines - self.degree + 1
        self.dknots = self.Tf/ (self.nKnots - 1)
    # end __init__
 
    # Evaluate the spline basis functions at time t:
    # This returns the values of d+1 spline basis functions, and the interval k such that t \in [tau_k, \tau_k+1] for spline knots \tau_i
    def eval(self, time):

        # Get interval index k s.t. t \in [t_k, t_k+1]
        k = int(time / self.dknots)   # this will round down to next smaller integer

        # Start with coefficient vector set to unit vector 1, 0, 0, 0...
        spline = []
        spline.append(1.0)

        # Recursive loop to update splines
        deltaKnots = self.dknots
        for i in range(1,self.degree+1):        # i = 1,2,...,degree
            spline.append(0.0)
            for r in range(i,0,-1):        # r = i, i-1, ..., 1
                coeff1 = (time - (k-i+r)*deltaKnots)  / ( (k+r)*deltaKnots - (k-i+r)*deltaKnots )
                coeff2 = ( (k+r+1)*deltaKnots - time) / ( (k+r+1)*deltaKnots - (k-i+r+1)*deltaKnots )
                spline[r] = coeff1 * spline[r-1] + coeff2 * spline[r]
            spline[0] = spline[0] * ((k+1)*deltaKnots - time) / ((k+1)*deltaKnots - (k-i+1)*deltaKnots)

        return spline, k
    # end eval(time)


def spline_test(degree, nSplines, Tfinal, deltax):
    print("Testing ", nSplines, " Bsplines of degree ", degree)

    # Init grid and splines
    n = int(Tfinal / deltax)
    xgrid = linspace(0.0, Tfinal, n+1)

    # Initialize splines
    nKnots = nSplines - degree + 1
    deltaKnots = Tfinal / (nKnots-1)
    spline = zeros((nSplines+1)*(n+1)).reshape(nSplines+1, n+1)
    splinebasis = BsplineBasis(nSplines, degree, Tfinal)

    # Loop over time domain and compute spline coefficients
    for i in range(len(xgrid)):

        time = xgrid[i]
        l = int(time / deltaKnots)
        spline[l:l+degree+1,i], k = splinebasis.eval(time)
        # print(spline[l:l+degree+1,i])

    # Plot
    for i in range(nSplines):
        plt.plot(xgrid, spline[i,:], label="B_"+str(i))
        # print(i, " ", spline[i,:])
    plt.legend()
    plt.show()

if __name__ == '__main__':
  import matplotlib.pyplot as plt

  # # Test now:
  # degree = 2
  # nSplines = 10
  # Tfinal = 1.0
  # deltax = 0.01
  # spline_test(degree, nSplines, Tfinal, deltax)
