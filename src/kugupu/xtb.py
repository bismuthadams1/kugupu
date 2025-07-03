from models_abc import CouplingModel
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Literal
from MDAnalysis.core.groups import AtomGroup
import MDAnalysis as mda
from MDAnalysis import AtomGroup
from rdkit.Chem import rdmolfiles
import tempfile



from .dimers import find_dimers



class XTB(CouplingModel):
    _name = 'xtb'

    def __init__(self, *, local: bool = True):
        super().__init__(local=local, server_id=server_id):
            if not self.local:
                if hasattr(self.server_id, "submit"):
                    self.client = self.server_id
                else:
                    raise ValueError(
                        "Please provide a dask client"
                        )
            else:
                self.client = None

def _atomgroup_to_xyz(
    atomlist: AtomGroup,
) -> str:
    """Convert an MDAnalysis Atomgroup to an xyz

    Parameters
    ----------
    atomlist: AtomGroup
        MDAnalysis AtomGroup to convert

    Returns
    -------
    
    """
    rdkit_mol = atomlist.convert_to.rdkit()
    xyz_block = rdmolfiles.MolToXYZBlock(rdkit_mol)

    return xyz_block

def _convert_to_model_format(
    fragments: List[AtomGroup],
    nn_cutoff: float
    ) -> Dict[tuple,str]:
    """
    Find all dimer pairs within nn_cutoff, and convert each pair into a Pymatgen Molecule.

    Parameters
    ----------

    Returns
    -------


    """

    for frag in fragments:
        mda.lib.mdamath.make_whole(frag)

    dimers = find_dimers(fragments, nn_cutoff)

    dimers_pymat: Dict[tuple, str] = {}
    for (i, j), ag_pair in dimers.items():
        mol = _atomgroup_to_xyz(ag_pair)
        dimers_pymat[(i,j)] = mol

    return dimers_pymat

def _compute_xtb_frame_from_fragments(
        fragments: List[AtomGroup],
        nn_cutoff: float,
        degeneracy: np.ndarray,
        state: str,  #implement soon      
) -> np.ndarray:
    dimers_dict = _convert_to_model_format(fragments, nn_cutoff)
    size = degeneracy.sum()
    H_frag = np.zeros((size, size))
    stops = np.cumsum(degeneracy)
    starts = np.r_[0, stops[:-1]]
    diag = np.arange(size)
    wave = dict()  # in OcelotML scenario, we just store a dummy

    all_mols = list(dimers_dict.values())
    predictions = _xtb_from_list(all_mols, ocelotml_model)

    for idx, ((i, j), mol) in enumerate(dimers_dict.items()):
        ix, iy = starts[i], stops[i]
        jx, jy = starts[j], stops[j]

        H_frag[diag[ix:iy], diag[ix:iy]] = predictions[idx]  # coupling(i→j)
        H_frag[diag[jx:jy], diag[jx:jy]] = predictions[idx]  # symmetric

        wave[i] = 0
        wave[j] = 0

    for i in (set(range(len(degeneracy))) - set(wave.keys())):
        ix, iy = starts[i], stops[i]
        single_mol = _atomgroup_to_pymatgen_molecule(fragments[i])
        e_i = predict_from_molecule(molecule=single_mol, model=ocelotml_model)
        H_frag[diag[ix:iy], diag[ix:iy]] = e_i

    return H_frag, None

def _xtb_from_list(
    dimers_list: List[str],
    mode: Literal['homo','lumo'],
    flavour: Literal['gfn1-XTB','gfn2-XTB','gfn'],
    threshold: float,
)-> tuple[int,]:
    
    flavour_dict ={
        'gfn1-XTB':''
    }

    try:
        flavour_cmd = flavour_dict[flavour]
    except KeyError:
        raise "Non-existent xtb flavour provided"

    
    for dimer in dimers_list:
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(dimer)

            cmd = f"xtb {fp.name}.xyz --dipro {} {}"



    
    

def _parse_xtb_outpus(
        
)