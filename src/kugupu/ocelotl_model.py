# ocelotl_model.py

import numpy as np
import MDAnalysis as mda
from typing import List, Dict, Optional, Tuple, Any
from MDAnalysis.core.groups import AtomGroup
from pymatgen.core.structure import Molecule as PymatgenMolecule
import pickle as pkl

from .models_abc import CouplingModel
from .dimers import find_dimers
from . import logger

from ocelotml import load_models, predict_from_list, predict_from_molecule

# Load your OcelotML weights once (e.g. “hh” model)
# Here we can add options for the lumo model too
ocelotml_model = load_models('hh')

class OcelotMLModel(CouplingModel):
    _name = "ocelotml"

    def __init__(self, *, local: bool = False, server_id: Optional["distributed.Client"] = None):
        super().__init__(local = local, server_id = server_id) 
        if not self.local:
            if hasattr(self.server_id, "submit"):
                self.client = self.server_id
            else:
                raise ValueError(
                    "Please provide a dask client"
                )
        else:
            self.client = None


    def __call_local__(
            self, 
            fragments: List[AtomGroup],
            nn_cutoff: float,
            degeneracy: np.ndarray,
            state: Optional[str] = 'homo' #this is where we can pick between models
            ) -> np.ndarray:
        """
        Build H_frag from scratch using OcelotML (predict_from_list/predict_from_molecule).
        Expected kwds: nn_cutoff (float), degeneracy (1D array), state unused here.
        """

        return _compute_ocelot_frame_from_fragments(
            fragments =fragments,
            nn_cutoff = nn_cutoff,
            degeneracy = degeneracy,
            state = state
        )
        

    def __call_remote__(
        self,
        top_pickle: pkl,
        traj_filename: str,
        frame_idx: int,
        nn_cutoff: float,
        degeneracy: np.ndarray,
        state: str,
    ) -> np.ndarray:
        """
        Remote/Dask path for a single frame.  We:
          1) Scatter `top_pickle` once (so workers can rebuild Universe).
          2) Submit one delayed task (`_dask_single_universe`) to compute H_frag on that worker.
          3) Return the resulting H_frag array.

        Expected keyword arguments (in **kwds):
          - nn_cutoff (float)
          - degeneracy (np.ndarray of ints)
          - state (str)
          - start, stop, step  [these are ignored here, because this is per‐frame]
        """

        u_worker = mda.Universe(top_pickle)
        u_worker.load_new(traj_filename)
        u_worker.trajectory[frame_idx]
        fragments = u_worker.atoms.fragments
        return self.__call_local__(fragments, nn_cutoff, degeneracy, state)

def _convert_to_model_format(fragments: List[AtomGroup], nn_cutoff: float) -> Dict[tuple,PymatgenMolecule]:
        """
        Find all dimer pairs within nn_cutoff, and convert each pair into a Pymatgen Molecule.
        """
        # nn_cutoff = kwds["nn_cutoff"]

        for frag in fragments:
            mda.lib.mdamath.make_whole(frag)

        dimers = find_dimers(fragments, nn_cutoff)

        dimers_pymat: Dict[tuple, PymatgenMolecule] = {}
        for (i, j), ag_pair in dimers.items():
            mol = _atomgroup_to_pymatgen_molecule(ag_pair)
            dimers_pymat[(i,j)] = mol

        return dimers_pymat

def _compute_ocelot_frame_from_fragments(
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
    predictions = predict_from_list(all_mols, ocelotml_model)

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

    return H_frag

def _atomgroup_to_pymatgen_molecule(atomgroup) -> dict[tuple, PymatgenMolecule]:
    """Takes a tuple of AtomGroup objects and turns them into a single pymatgen object

    Parameters
    ----------
    atomgroup: AtomGroup

    Returns
    -------
    mol: Molecule

    
    """
    if isinstance(atomgroup, tuple): 
        atomgroup = sum(atomgroup[-1])

    elements = atomgroup.names
    coords = atomgroup.positions  # NumPy array of shape (n_atoms, 3)
   
    return  PymatgenMolecule(elements, coords)