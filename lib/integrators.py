"""
SGD and GDN exact integrators
"""

import numpy as np

class Integrator:
    """
    Parent integrator class for the loss (1/(2*batchsize))*\sum_i ||epsilon_i - q(w)*x_i||^2, where:
    * x_i and epsilon_i are independent 1-dimensional Gaussian random variables
    * w is a d-dimensional parameter vector
    * q is an arbitrary scalar function of w
    """
    def __init__(self, std_epsilon, lr, q, grad_q, w_init, batchsize, seed, pbc=False):
        """
        std_epsilon: standard deviation of the noise (nb: the x_i variables have a fixed variance of 1)
        lr: learning rate
        q: model
        grad_q: gradient of the model
        w_init: initial value of the parameter vector
        batchsize: batch size
        seed: seed used to generate data
        pbc: if True, dynamics is on the unit circle; if False, dynamics is on the real line
        """
        self.lr = lr
        self.q = q
        self.grad_q = grad_q
        self.nb = batchsize
        self.w = [w_init]
        self.state = np.random.RandomState(seed=seed)
        self.std_epsilon = std_epsilon
        self.pbc = pbc

        self.f = self.std_epsilon/np.sqrt(2.)
        self.sqb = np.sqrt(self.nb)

    def evolve(self, nstep):
        wc = self.w[-1]
        for _ in range(nstep):
            wc = self.update(wc)
            self.w.append(wc)
 
            # if np.isnan(wc): # stop current run if 
                # break

class SGD(Integrator):
    """
    SGD (exact) integrator
    """
    def update(self, w_old):
        xb = self.state.normal(size=self.nb)
        yb = self.state.normal(size=self.nb, scale=self.std_epsilon)
        
        xi_xx = np.mean(xb*xb)
        xi_xy = np.mean(xb*yb)
        grad0 = (xi_xx * self.q(w_old) - xi_xy) * self.grad_q(w_old)
        
        w_new = w_old - self.lr*grad0
        
        if self.pbc:
            w_new = w_new % 1.
            
        return w_new
            

class GDN(Integrator):
    """
    GD + Gaussian noise integrator
    I.e., replaces SGD minibatch random variables by Gaussian random variables
    """        
    def update(self, w_old):
        xi1, xi2 = self.state.normal(scale=np.sqrt(2), size=2)
        
        grad = self.q(w_old) + (self.q(w_old)*xi1 - self.f*xi2)/self.sqb
        grad *= self.grad_q(w_old)
        w_new = w_old - self.lr*grad
        
        if self.pbc:
            w_new = w_new % 1.
        
        return w_new
    