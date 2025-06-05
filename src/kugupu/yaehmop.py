import numpy as np
import MDAnalysis as mda
from typing import List, Any, Optional
from MDAnalysis.core.groups import AtomGroup
from pymatgen.core.structure import Molecule as PymatgenMolecule

from .models_abc import CouplingModel
from .dimers import find_dimers
from . import logger
from ._yaehmop import run_dimer, run_fragment
from ._hamiltonian_reduce import find_psi



class YaehmopModel(CouplingModel):
    _name = "yaehmop"

    def __init__(self, *, local: bool = False, server_id: Optional[Any] = None):
        """
        If local=True, everything runs in‐process. Otherwise, server_id must be
        a dask.distributed.Client, which we will use for remote submission.
        """
        super().__init__(local=local, server_id=server_id)
        if not self.local:
            if hasattr(self.server_id, "submit"):
                self.client = self.server_id
            else:
                raise ValueError("Remote mode requires a Dask Client.")
        else:
            self.client = None

    def __call_local__(
        self,
        fragments: List[AtomGroup],
        nn_cutoff: float,
        degeneracy: np.ndarray,
        state: str = "homo",
    ) -> np.ndarray:
        """
        Local path: compute a single‐frame coupling matrix entirely in‐process.
        """
        return _compute_yaehmop_frame_from_fragments(
            fragments, nn_cutoff, degeneracy, state
        )

    def __call_remote__(
        self,
        top_pickle: Any,
        traj_filename: str,
        frame_idx: int,
        nn_cutoff: float,
        degeneracy: np.ndarray,
        state: str = "homo",
    ) -> np.ndarray:
        """
        Remote path: submit exactly one task to the Dask cluster, which will
        rebuild the Universe on a worker and call the same local helper.
        """
        if self.client is None:
            raise RuntimeError("No Dask client available. Construct with local=False.")

        future = self.client.submit(
            _compute_yaehmop_frame_from_universe,
            top_pickle,
            traj_filename,
            frame_idx,
            nn_cutoff,
            degeneracy,
            state,
        )
        return future.result()
    
def _compute_yaehmop_frame_from_universe(
    top_pickle: Any,
    traj_filename: str,
    frame_idx: int,
    nn_cutoff: float,
    degeneracy: np.ndarray,
    state: str,
) -> np.ndarray:
    """
    Dask worker entrypoint for one frame:
      1) Rehydrate Universe from pickled topology
      2) Load trajectory, jump to frame_idx
      3) Extract fragments = u_worker.atoms.fragments
      4) Delegate to _compute_yaehmop_frame_from_fragments
    """
    u_worker = mda.Universe(top_pickle)
    u_worker.load_new(traj_filename)
    u_worker.trajectory[frame_idx]
    fragments = u_worker.atoms.fragments

    return _compute_yaehmop_frame_from_fragments(
        fragments, nn_cutoff, degeneracy, state
    )

def _compute_yaehmop_frame_from_fragments(
    fragments: List[AtomGroup],
    nn_cutoff: float,
    degeneracy: np.ndarray,
    state: str,
) -> np.ndarray:
    """
    Exactly the logic of _single_frame, but factored out as a helper.
    Given:
      - fragments: List[AtomGroup] for one frame
      - nn_cutoff: nearest‐neighbour cutoff
      - degeneracy: 1D array of degeneracies
      - state: 'homo' or 'lumo'
    Returns
      - H_frag: 2D numpy array of shape (sum(degeneracy), sum(degeneracy))
    """
    # (a) Make each fragment whole under periodic boundaries
    for frag in fragments:
        mda.lib.mdamath.make_whole(frag)

    # (b) Find all dimer pairs within nn_cutoff
    dimers = find_dimers(fragments, nn_cutoff)

    # (c) Prepare H_frag with correct size
    size = int(degeneracy.sum())
    H_frag = np.zeros((size, size), dtype=float)

    stops = np.cumsum(degeneracy).astype(int)
    starts = np.r_[0, stops[:-1]].astype(int)
    diag_idx = np.arange(size)

    # (d) Temporary storage for wavefunctions
    wave = {}

    # (e) Loop over each dimer (i,j) in sorted order
    for (i, j), ag_pair in sorted(dimers.items()):
        ix, iy = starts[i], stops[i]
        jx, jy = starts[j], stops[j]

        logger.debug(f"Yaehmop: computing dimer {i}-{j}")
        Hij, frag_i, frag_j = run_dimer(ag_pair)

        # Fragment i
        if i in wave:
            psi_i = wave[i]
            e_i = None  # energy already on diagonal
        else:
            e_i, psi_i = find_psi(frag_i[0], frag_i[1], frag_i[2], state, degeneracy[i])
            H_frag[diag_idx[ix:iy], diag_idx[ix:iy]] = e_i
            wave[i] = psi_i

        # Fragment j
        if j in wave:
            psi_j = wave[j]
            e_j = None
        else:
            e_j, psi_j = find_psi(frag_j[0], frag_j[1], frag_j[2], state, degeneracy[j])
            H_frag[diag_idx[jx:jy], diag_idx[jx:jy]] = e_j
            wave[j] = psi_j

        # Off‐diagonal: coupling = |ψ_iᵀ·Hij·ψ_j|
        coupling_val = abs(psi_i.T.dot(Hij).dot(psi_j))
        H_frag[ix:iy, jx:jy] = coupling_val
        H_frag[jx:jy, ix:iy] = coupling_val

    # (f) Handle any fragment not yet seen (lone fragments)
    unseen = set(range(len(degeneracy))) - set(wave.keys())
    for i in unseen:
        ix, iy = starts[i], stops[i]
        logger.debug(f"Yaehmop: computing lone fragment {i}")
        H_mat, S_mat, ele = run_fragment(fragments[i])
        e_i, psi_i = find_psi(H_mat, S_mat, ele, state, degeneracy[i])
        H_frag[diag_idx[ix:iy], diag_idx[ix:iy]] = e_i
        # no need to store psi for lone fragments

    return H_frag