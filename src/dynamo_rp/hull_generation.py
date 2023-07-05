import numpy as np
from dynamo_rp import rp_utility as ut


def write_list_of_list_vtx(file_name, a_list):
    """
    Write a list of list to a file. Each list is a line in the file.
    Parameters:
      file_name: file name
      a_list: a list of list
    Return:
      None
    """
    import os

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as fp:
        lines = [",".join([str(s) for s in item]) for item in a_list]
        fp.write("\n".join(lines))


def get_hull_points_from_pdb(module_chain, pdb_fname, chain_label="a"):
    """
    Get the hull points from pdb file
    Parameters:
      module_chain: a list of modules present e.g. ['NcapD4', 'D4', 'D4', 'D4', 'CcapD4']
      pdb_fname: pdb filename
    Optional:
      chain_label: label of chain in pdb file
    """
    import Bio.PDB as bp

    parser = bp.PDBParser()
    structure = parser.get_structure("rp", pdb_fname)
    starts_ends = np.concatenate(
        ut.get_helical_residue_location_for_chain(module_chain)
    )
    ar = np.array([res["CA"].coord for res in structure[0][chain_label]])
    starts = []
    ends = []
    for i, j in starts_ends:
        starts.append(ar[i - 1])
        ends.append(ar[j - 1])
    return np.array(starts), np.array(ends)
