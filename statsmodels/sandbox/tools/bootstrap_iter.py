'''blocks bootstrap (without overlap) and moving blocks bootstrap (with overlap)

circular wrapping, last block might include first observation

brief intro http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/ebooks/html/csa/node132.html#SECTION019142000000000000000
'''


import numpy as np


def it_boots_moving_blocks(nobs, block_length, truncate=False):
    '''generator for moving block bootstrap

    Notes
    -----
    This uses wrapping from the end to the beginning of the data series.
    If truncate is True, then the returned index is valid for the original
    data series.
    If truncate is False, then the returned index array has a largest possible
    index equal to ``nobs + block_length - 2``. It needs to index into an
    array that has the     first ``block_length - 1`` observations concatenated
    to the end.

    #TODO: reverse indexing so we have negative indices for automatic wrapping
    '''
    n_blocks = int(np.ceil(nobs * 1. / block_length))
    idx0 = np.cumsum(np.ones((n_blocks, block_length), int), 1) - 1

    while True:
        #start = np.random.randint(0, nobs, size=n_blocks)
        #wrap with negative
        start = np.random.randint(0, nobs, size=n_blocks) - block_length + 1

        #moving blocks, with overlap
        idx = (idx0 + start[:,None]).ravel()
        if truncate:
            #reindex wrapped observations
            mask = idx >= nobs
            idx[mask] -= nobs
        yield idx[:nobs]

def it_boots_blocks(nobs, block_length, truncate=True):
    #BUG: last block if not full, is filled with beginning, not truncated
    #     fixed when truncate=True
    n_blocks = int(np.ceil(nobs * 1. / block_length))
    idx0 = np.cumsum(np.ones((n_blocks, block_length), int), 1) - 1
    idx0 += np.arange(0, nobs, block_length)[:,None]
    #idx0 += np.arange(nobs, -1, -block_length)[:,None]

    while True:
        start = np.random.randint(0, n_blocks, size=n_blocks)

        #blocks, without overlap, except for last block with first observations
        idx = idx0[start, :].ravel()
        if truncate:
            #remove wrapped observations
            mask = idx < nobs
            idx = idx[mask]
            n_iter = 0   #emergency break
            while len(idx) < nobs and n_iter < 10:
                #this is inefficient
                n_miss = int(np.ceil((nobs - len(idx)) *1./ block_length) * 2)
                start = np.random.randint(0, n_blocks, size=n_miss)
                idx = np.concatenate((idx, idx0[start, :].ravel()))
                mask = idx < nobs
                idx = idx[mask]
                n_iter += 1
        yield idx[:nobs]
