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


def get_rp_helices(rp, conf, folder, start_end=(None, None)):
    """
    Get the helices for a given rp and conf in a given folder.
    Folder structure must be: 
    folder
    ├── rp1
    │   ├── rp1_0001.vtx
    The format of the vtx file can be seen in
    hull_generation.write_list_of_list_vtx
    Parameters:
      rp (str): rp name
      conf (int): conf number
      folder (str): folder name
    Optional:
      start_end (tuple): start and end helix index
    Returns:
      a numpy array of shape (num_helices, 2, 3)
    """
    file_name = f"{folder}/{rp}/{rp}_{conf:04}.vtx"
    point_list = hg.read_list_of_list_vtx(file_name)
    num_helices = len(point_list) // 2
    out = []
    for i in range(num_helices):
        u = np.array(point_list[i][  1: ])
        l = np.array(point_list[i + num_helices][1:])
        out.append(np.array([u, l]))

    coords = np.array(out)

    return coords[start_end[0]:start_end[1], :, :]


def get_rp_helices_all(rp, conf_range, folder, array_out=True, **kwargs):
    """
    Get the helices for a given rp and confrange in a given folder.
    Folder structure must be: 
    folder
    ├── rp1
    │   ├── rp1_0001.vtx
    Parameters:
      rp (str): rp name
      conf_range (range): range of confs
      folder (str): folder name
    Optional:
      array_out (bool): return as numpy array
      **kwargs: passed to get_rp_helices
    Returns:
      a numpy array of shape (num_confs, num_helices, 2, 3)
    """
    out = []
    for conf in conf_range:
        coords = get_rp_helices(rp, conf, folder, **kwargs)
        out.append(coords)
    if array_out:
        return np.array(out)
    else:
        return out


def split_hels_into_modules(rp, hels):
    """
    Split the helices into modules based on the rp.
    Assumes the helices are in the order of the rp,
    and that caps are included. Uses the rp_module_list database.
    Parameters:
        rp (str): rp name
        hels (np.array): helices of shape (num_helices, 2, 3)
    Returns:
        a tuple of helices for each module in the rp.
        (hels_a_cap, hels_a, hels_b, hels_c, hels_c_cap)
    """
    hel_idxs = np.cumsum([pm.module_num_helices[mod]
                          for mod in pm.rp_module_list[rp]])
    hels_a_cap = hels[:, 0:hel_idxs[1]]
    hels_a = hels[:, hel_idxs[0]:hel_idxs[1]]
    hels_b = hels[:, hel_idxs[1]:hel_idxs[2]]
    hels_c = hels[:, hel_idxs[2]:hel_idxs[3]]
    hels_c_cap = hels[:, hel_idxs[2]:]
    return hels_a_cap, hels_a, hels_b, hels_c, hels_c_cap


