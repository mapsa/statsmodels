'''Example for block and moving block bootstrap generator

'''


import numpy as np
from statsmodels.sandbox.tools.bootstrap_iter import (it_boots_moving_blocks,
                                                    it_boots_blocks)


seed = np.random.randint(1000000)
#seed = 893991  #to show max idx is 21, 20
#seed = 487061
print(seed)
np.random.seed(seed)

nobs = 20
x = np.arange(nobs)

block_length = 3   #block length
x2 = np.concatenate((x, x[:block_length - 1]))

b_iter = it_boots_moving_blocks(nobs, block_length)

bidx = b_iter.next()
#print(start)
print(bidx)
print(x2[bidx])

print('starting loop')
for _ in range(5):
    bidx = b_iter.next()
    print(bidx),
    print(len(bidx), max(bidx))

b2_iter = it_boots_blocks(nobs, block_length)
print('starting loop')
for _ in range(5):
    bidx = b2_iter.next()
    print(bidx),
    print(len(bidx), max(bidx))


rvs_idx = np.array([x[b_iter.next()] for _ in range(10000)])
print('bincount all indices')
print(np.bincount(rvs_idx.ravel()))
print('mean index at each position')
print(rvs_idx.mean(0))


rvs_idx2 = np.array([b2_iter.next() for _ in range(10000)])
print('bincount all indices')
print(np.bincount(rvs_idx2.ravel()))
print('mean index at each position')
print(rvs_idx2.mean(0))
