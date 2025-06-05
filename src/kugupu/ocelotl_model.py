# ocelotl_model.py
import dask.delayed
import numpy as np
from tqdm import tqdm
import MDAnalysis as mda
from typing import List, Dict, Any
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis import Universe
from pymatgen.core.structure import Molecule as PymatgenMolecule
from typing import List, Dict, Any, Optional, Tuple

from .models_abc import CouplingModel
from .dimers import find_dimers
from . import logger

from ocelotml import load_models, predict_from_list, predict_from_molecule

# Load your OcelotML weights once (e.g. “hh” model)
ocelotml_model = load_models('hh')


class OcelotMLModel(CouplingModel):
    _name = "ocelotml"

    def __init__(self, *, local: bool = False, server_id: Optional[str] = None):
        super().__init__(local = local, server_id = server_id) 
        if not self.local:
            if hasattr(self.server_id, "submit"):
                self.client = self.server_id
            else:
                from dask.distributed import Client
                self.client = Client()
        else:
            self.client = None


    def _convert_to_model_format(self, fragments: List[AtomGroup], **kwds) -> List[PymatgenMolecule]:
        """
        Turn every dimer (a tuple of two AtomGroup objects) into a single
        Pymatgen Molecule.  Just return the list of Molecule objects.
        We still need nn_cutoff (float) to know which dimers to form.
        """
        nn_cutoff = kwds["nn_cutoff"]

        for frag in fragments:
            mda.lib.mdamath.make_whole(frag)

        dimers = find_dimers(fragments, nn_cutoff)

        dimers_pymat: Dict[tuple, PymatgenMolecule] = {}
        for (i, j), ag_pair in dimers.items():
            mol = self._convert_to_model_format(ag_pair)
            dimers_pymat[(i, j)] = mol

        return dimers_pymat  

    def __call_local__(
            self, 
            fragments: List[AtomGroup],
            **kwds) -> np.ndarray:
        """
        Build H_frag from scratch using OcelotML (predict_from_list/predict_from_molecule).
        Expected kwds: nn_cutoff (float), degeneracy (1D array), state unused here.
        """
        degeneracy = kwds["degeneracy"]

        dimers_dict = self._convert_to_model_format(fragments,  **kwds)

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
            single_mol = self._convert_to_model_format(fragments[i])
            e_i = predict_from_molecule(molecule=single_mol, model=ocelotml_model)
            H_frag[diag[ix:iy], diag[ix:iy]] = e_i

        return H_frag

    def __call_remote__(
        self,
        top_pickle: Any,
        traj_filename: str,
        frame_idx: int,
        **kwds: Any
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
        if self.client is None:
            raise RuntimeError(
                "No Dask client available. Construct this model with local=False."
            )

        nn_cutoff = kwds["nn_cutoff"]
        degeneracy = kwds["degeneracy"]
        state = kwds["state"]

        future_top = self.client.scatter(top_pickle, broadcast=True)

        import dask

        delayed_task = dask.delayed(_dask_single_universe)(
            future_top,
            traj_filename,
            frame_idx,
            nn_cutoff,
            degeneracy,
            state,
            self,  # pass the model instance so the worker can call __call_local__
        )

        future = self.client.compute(delayed_task)
        return future.result()

    

def _dask_single_universe(top, trj, frame, nn_cutoff, degeneracy, state, model_instance: OcelotMLModel):
    """Dask helper function for calculating a single frame

    Parameters
    ----------
    top : pickle
    pickled MDAnalysis Topology, usually broadcasted to workers
    trj : str
    filename to the trajectory file
    frame : int
    index of the frame to analyse
    nn_cutoff, degeneracy, state
    same as for _single_frame

    Reheats the MDAnalysis Universe, loads correct frame then calls _single_frame
    """
    # load the Universe
    u_worker = mda.Universe(top)
    u_worker.load_new(trj)
    # select correct frame
    u_worker.trajectory[frame]

    fragments = u_worker.atoms.fragments

    return model_instance.__call_local__(
       fragments = fragments,
       nn_cutoff = nn_cutoff,
       degeneracy = degeneracy,
       state = state
    )



    # def _convert_to_model_format(self, atomgroup: AtomGroup) -> PymatgenMolecule:
    #     """
    #     Turn a single AtomGroup (or a tuple of two AtomGroups) into a Pymatgen Molecule.
    #     If it’s a tuple (e.g. a dimer), sum the last AtomGroup in the tuple.
    #     """
    #     if isinstance(atomgroup, tuple):
    #         # e.g. atomgroup = (idx_pair, (AtomGroup1, AtomGroup2))
    #         # we only want the actual AtomGroup list at index [-1]
    #         atomgroup = atomgroup[-1]  # this flattens to a single AtomGroup
    #     elements = atomgroup.names
    #     coords = atomgroup.positions
    #     return PymatgenMolecule(elements, coords)


