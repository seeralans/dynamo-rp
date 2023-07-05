import numpy as np
from dynamo_rp import parameters as pm


def get_res_counts_for_chain(chain):
    """
    Computes the cumulative sum of the number of residues in each module in the chain.
    Parameters:
      chain: a list of modules present e.g. ['NcapD4', 'D4', 'D4', 'D4', 'CcapD4']
    Returns:
      a list of cumulative sum of the number of residues in each module in the chain.
    """
    return list(np.array([pm.module_lengths[mod] for mod in chain]).cumsum())


def get_helical_counts_for_chain(chain):
    """
    Gets the cumulative sum of the number of helices in each module in the chain.
    Parameters:
      chain: a list of modules present e.g. ['NcapD4', 'D4', 'D4', 'D4', 'CcapD4']
    Returns:
      a list of cumulative sum of the number of helices in each module in the chain.
    """
    return list((np.array([pm.module_num_helices[mod] for mod in chain]).cumsum()))


# TODO FIX FLIPPING
def get_helical_residue_location_for_chain(chain):
    """
    Gets the helical residue location for each module in the chain.
    Parameters:
      chain: a list of modules present e.g. ['NcapD4', 'D4', 'D4', 'D4', 'CcapD4']
    Returns:
      a list of lists of helical residue locations for each module in the chain.
    """
    res_count = [0] + get_res_counts_for_chain(chain)
    switch = lambda k, pair: pair if k % 2 == 0 else pair[::-1]
    helices = [
        [
            [idx + res_count[i] for idx in switch(k, pair)]
            for k, pair in enumerate(pm.module_helical_start_ends[mod])
        ]
        for i, mod in enumerate(chain)
    ]
    return helices
