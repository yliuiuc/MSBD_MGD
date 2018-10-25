"""
Manifold Gradient Descent for Multichannel Sparse Blind Deconvolution
- 2D image deconvolution
- Super-resolution fluorescence microscopy example

@author: Yanjun Li (Local)
"""

import numpy as np
from scipy import linalg as la
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.fft import fft2, ifft2
from PIL import Image


def gradientL(h, Y, R, N):
    Ch = np.real(ifft2(fft2(Y) * R[None, :, :] * fft2(h)[None, :, :]))
    G = -1/N * np.real(ifft2(np.sum((fft2(Y) * R[None, :, :]).conj() 
        * fft2(Ch**3), axis=0)))
    RG = G - h * (h.ravel().dot(G.ravel()))
    return RG

def mgd(Y, N, n, theta, gamma=1e-1, numIter=100, h0=None):
    
    Cov = np.mean(np.abs(fft2(Y))**2, axis=0) / (n**2 * theta)
    R = 1 / np.sqrt(Cov)
    if h0 is None:
        h = np.zeros((n, n))
        h[1, 1] = 1
    else:
        h = h0
    h /= la.norm(h)
    
    for idxIter in range(numIter):
        print('Iter # {}'.format(idxIter))
        hnLh = gradientL(h, Y, R, N)
        h -= gamma * hnLh
        h /= la.norm(h)
    
    return h, R


if __name__ == "__main__":

    ######## Super-Resolution Fluorescence Microscopy Example ########

    # Load and synthesize data
    DATA =loadmat('my_data.mat')
    f = DATA['PSF_true']
    Xs = DATA['X']
    n, _, N = Xs.shape
    X = np.zeros((N, n, n))
    for i in range(N):
        X[i, :, :] = Xs[:, :, i]
    x = np.mean(X, axis=0)
    Y = np.real(ifft2(fft2(f)[None, :, :] * fft2(X)))
    y = np.mean(Y, axis=0)
    theta = 0.005
    
    # Recover using MGD
    h, R = mgd(Y, N, n, theta, gamma=1e-2, numIter=100)
    fhat = np.real(ifft2(1 / (R * fft2(h))))
    plt.figure()
    plt.axis('off')
    plt.imshow(fhat, cmap='gray')
    
    Xhat = np.real(ifft2(fft2(Y) * R[None, :, :] * fft2(h)[None, :, :]))
    xhat = np.mean(Xhat, axis=0)
    plt.figure()
    plt.imshow(xhat, cmap='gray')
    
    
    
    ######## Save Results ########
    
    # Save true image (average image)
    out = x
    out = (out - np.min(out)) / (np.max(out) - np.min(out))
    cmap = plt.get_cmap(name='coolwarm')
    out = cmap(out)
    img255 = np.uint8(out * 255)
    imgres = Image.fromarray(img255)
    imgres.save('micro_true2.png')
    
    # Save true image (examples of frames)
    array = X
    tmpY = array[0, :, :]
    Y0 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[1, :, :]
    Y1 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[99, :, :]
    Y2 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[100, :, :]
    Y3 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    xeg = 0.5 * np.ones((n * 2 + 5, n * 2 + 5))
    xeg[:n, :][:, :n] = Y0
    xeg[:n, :][:, -n:] = Y1
    xeg[-n:, :][:, :n] = Y2
    xeg[-n:, :][:, -n:] = Y3
    cmap = plt.get_cmap(name='coolwarm')
    xeg = cmap(xeg)
    imgxeg = np.uint8(xeg * 255)
    imgxeg = Image.fromarray(imgxeg)
    imgxeg.save('micro_true1.png')  
    
    # Save blurred image (average image)
    ri, ci = np.unravel_index(np.argmax(f), f.shape)
    vec = np.zeros(f.shape)
    vec[ri, ci] = 1
    yshift = np.real(ifft2(fft2(y) / fft2(vec)))
    out = yshift
    out = (out - np.min(out)) / (np.max(out) - np.min(out))
    cmap = plt.get_cmap(name='coolwarm')
    out = cmap(out)
    img255 = np.uint8(out * 255)
    imgres = Image.fromarray(img255)
    imgres.save('micro_blur2.png')  

    # Save blurred image (examples of frames)
    ri, ci = np.unravel_index(np.argmax(f), f.shape)
    vec = np.zeros(f.shape)
    vec[ri, ci] = 1
    Yshift = np.real(ifft2(fft2(Y) / fft2(vec)[None, :, :]))
    array = Yshift
    tmpY = array[0, :, :]
    Y0 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[1, :, :]
    Y1 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[99, :, :]
    Y2 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[100, :, :]
    Y3 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    xeg = 0.5 * np.ones((n * 2 + 5, n * 2 + 5))
    xeg[:n, :][:, :n] = Y0
    xeg[:n, :][:, -n:] = Y1
    xeg[-n:, :][:, :n] = Y2
    xeg[-n:, :][:, -n:] = Y3
    cmap = plt.get_cmap(name='coolwarm')
    xeg = cmap(xeg)
    imgxeg = np.uint8(xeg * 255)
    imgxeg = Image.fromarray(imgxeg)
    imgxeg.save('micro_blur1.png')

    # Save recovered image (average image)
    # with sign and shift correction
    vec = np.real(ifft2(fft2(f) * R * fft2(h)))
    thres = 0.9 * np.max(vec)
    vec[vec < thres] = 0
    vec[vec >= thres] = 1    
    xhatshift = np.real(ifft2(fft2(xhat) / fft2(vec)))
    out = xhatshift
    out[out < 0] = 0
    out = (out - np.min(out)) / (np.max(out) - np.min(out))
    cmap = plt.get_cmap(name='coolwarm')
    out = cmap(out)
    img255 = np.uint8(out * 255)
    imgres = Image.fromarray(img255)
    imgres.save('micro_recon2.png')   
    
    # Save recovered image (examples of frames)
    # with sign and shift correction
    vec = np.real(ifft2(fft2(f) * R * fft2(h)))
    thres = 0.9 * np.max(vec)
    vec[vec < thres] = 0
    vec[vec >= thres] = 1   
    Xhatshift = np.real(ifft2(fft2(Xhat) / fft2(vec)[None, :, :]))
    array = Xhatshift
    array[array < 0] = 0
    tmpY = array[0, :, :]
    Y0 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[1, :, :]
    Y1 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[99, :, :]
    Y2 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    tmpY = array[100, :, :]
    Y3 = (tmpY - np.min(tmpY)) / (np.max(tmpY) - np.min(tmpY))
    xeg = 0.5 * np.ones((n * 2 + 5, n * 2 + 5))
    xeg[:n, :][:, :n] = Y0
    xeg[:n, :][:, -n:] = Y1
    xeg[-n:, :][:, :n] = Y2
    xeg[-n:, :][:, -n:] = Y3
    cmap = plt.get_cmap(name='coolwarm')
    xeg = cmap(xeg)
    imgxeg = np.uint8(xeg * 255)
    imgxeg = Image.fromarray(imgxeg)
    imgxeg.save('micro_recon1.png')    


    # Save true kernel
    out = f
    out = (out - np.min(out)) / (np.max(out) - np.min(out))
    sx, sy = np.meshgrid(np.arange(64), np.arange(64))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sx, sy, out, cmap=cm.coolwarm)
    ax.set_axis_off()
    fig.savefig('micro_k_true.png', bbox_inches='tight')
    
    # Save recovered kernel
    # with sign and shift correction
    vec = np.real(ifft2(fft2(f) * R * fft2(h)))
    thres = 0.9 * np.max(vec)
    vec[vec < thres] = 0
    vec[vec >= thres] = 1 
    fhatshift = np.real(ifft2(fft2(vec) / (R * fft2(h))))
    out = fhatshift
    out = (out - np.min(out)) / (np.max(out) - np.min(out))
    sx, sy = np.meshgrid(np.arange(64), np.arange(64))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sx, sy, out, cmap=cm.coolwarm)
    ax.set_axis_off()
    fig.savefig('micro_k_recon.png', bbox_inches='tight')