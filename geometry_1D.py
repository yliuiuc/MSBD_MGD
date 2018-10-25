"""
Manifold Gradient Descent for Multichannel Sparse Blind Deconvolution
- 1D deconvolution
- Phase transition plot

@author: Yanjun Li
"""

import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib2tikz import save as tikz_save

# Compute the gradient of the loss function at h
def gradientL(h, C, N):    
    Ch = C.dot(h)
    G = -1/N * C.T.dot(Ch**3)
    RG = G - h * (h.dot(G))
    return RG

# Manifold Graident Descent algorithm for MSBD
# Input:
# - Cy: Circulant matrix
# - N: Number of channels
# - n: Length of signal
# - gamma: Step size
# - numIter: Number of iterations
# - h0: Initial estimate of inverse filter h (optional)

def mgd(Cy, N, n, theta, gamma=1e-1, numIter=100, h0=None):
    Cov = Cy.T.dot(Cy) / theta / n / N
    U, S, _ = la.svd(Cov)
    R = U.dot(np.diag(1 / np.sqrt(S))).dot(U.T)
    C = Cy.dot(R)    
    if h0 is None:
        h = np.random.randn(n)
        h /= la.norm(h)
    else:
        h = h0
        
    for idxIter in range(numIter):
        hnLh = gradientL(h, C, N)
        h -= gamma * hnLh
        h /= la.norm(h)
    
    return h, R
        

def randRun(N, n, theta, rseed):
    np.random.seed(rseed)
    sigma = 0.1 * np.sqrt(n * theta) # noisy
    f = np.random.randn(n)
    Cf = la.circulant(f)
    Cx = np.zeros((n * N, n))
    Cn = np.zeros((n * N, n))
    for i in range(N):
        uniform = np.random.rand(n, 2)
        xx = ((uniform[:, 0] < theta).astype(np.float) 
            * ((uniform[:, 1] > 0.5).astype(np.float) * 2 - 1))
        Cx[(i*n):((i+1)*n), :] = la.circulant(xx)
        Cn[(i*n):((i+1)*n), :] = la.circulant(sigma * np.random.randn(n))
    Cy = Cx.dot(Cf) + Cn
    
    h, R = mgd(Cy, N, n, theta, gamma=1e-1, numIter=100)
    vec = Cf.dot(R.dot(h))
    res = la.norm(vec, ord=np.inf) / la.norm(vec)
    
    return res


def randParam(N, n, theta, rnum):
    for rseed in range(rnum):
        yield N, n, theta, rseed

if __name__ == "__main__":


    ######## Phase transition for Manifold Gradient Descent ########
    
    Nnum = 8
    nnum = 8
    rnum = 100
    Nvec = 32 * np.arange(1, Nnum+1)
    nvec = 32 * np.arange(1, nnum+1)
    rvec = 2018 + np.arange(rnum)
    theta = 0.1
    filename = 'MGD-N_vs_n-Theta_0.1-SNR_20dB'
    
    RES = np.zeros((Nnum, nnum, rnum))
    for Nidx in range(Nnum):
        N = Nvec[Nidx]
        for nidx in range(nnum):
            n = nvec[nidx]
            print("N: {0}, n: {1}".format(N, n))
            param = randParam(N, n, theta, rnum)
            with Pool(7) as pool:
                res = pool.starmap(randRun, param)
            for ridx in range(rnum):
                RES[Nidx, nidx, ridx] = res[ridx]
            np.save(filename + '.npy', RES)
            
    RES = np.load(filename + '.npy')
    RES2 = np.mean((RES > 0.95).astype(np.float), axis=2) 
    plt.figure()
    plt.imshow(RES2, cmap='gray')
    tikz_save(filename + '.tex')
    

    plt.figure()
    RES = np.load(filename + '.npy')
    RES2 = np.mean((RES > 0.95).astype(np.float), axis=2) 
    plt.imshow(RES2, cmap='gray')
    tikz_save(filename + '.tex')