import spams

import numpy as np

import time



np.random.seed(0)

print ("test lasso weighted")

X = np.asfortranarray(np.random.normal(size=(64,10000)))

X = np.asfortranarray(X / np.tile(np.sqrt((X*X).sum(axis=0)),(X.shape[0],1)),dtype=float)

D = np.asfortranarray(np.random.normal(size=(64,256)))

D = np.asfortranarray(D / np.tile(np.sqrt((D*D).sum(axis=0)),(D.shape[0],1)),dtype=float)

param = { 'L' : 20,

    'lambda1' : 0.15, 'numThreads' : 8, 'mode' : spams.PENALTY}

W = np.asfortranarray(np.random.random(size = (D.shape[1],X.shape[1])),dtype=float)

tic = time.time()

alpha = spams.lassoWeighted(X,D,W,**param)

tac = time.time()

t = tac - tic

non_zero = []

for col in alpha.T:

    non_zero.append(col.nnz)

print ('Shape Output Matrix:', alpha.shape)

print ('Min non-zeros of %d columns: %d'%(alpha.shape[1], np.min(non_zero)))

print ('Max non-zeros of %d columns: %d'%(alpha.shape[1], np.max(non_zero)))

print ("%f signals processed per second\n" %(float(X.shape[1]) / t))
