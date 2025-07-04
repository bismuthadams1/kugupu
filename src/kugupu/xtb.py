from .models_abc import CouplingModel
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Literal
from MDAnalysis.core.groups import AtomGroup
import MDAnalysis as mda
from MDAnalysis import AtomGroup
from rdkit.Chem import rdmolfiles
import tempfile
import subprocess
import os
import re
from tqdm import tqdm
import pickle as pkl
from MDAnalysis.topology.guessers import guess_types

from ._yaehmop import shift_dimer_images

from . import logger

from .dimers import find_dimers

class XTBError(Exception):
    """Custom exception for xTB-related errors."""
    pass


class XTB(CouplingModel):
    _name = 'xtb'

    def __init__(self, *, local: bool = True, server_id: Optional["distributed.Client"] = None, xtb_model = 'gfn1-XTB'):
        super().__init__(local=local, server_id=server_id)
        if not self.local:
            if hasattr(self.server_id, "submit"):
                self.client = self.server_id
            else:
                raise ValueError(
                    "Please provide a dask client"
                    )
        else:
            self.client = None
        self.xtb_model = xtb_model
    
    def __call_local__(
            self, 
            fragments: List[AtomGroup],
            nn_cutoff: float,
            degeneracy: np.ndarray,
            state: Optional[str] = 'homo', #this is where we can pick between models
            ) -> np.ndarray:
        """
        Build H_frag from scratch using OcelotML (predict_from_list/predict_from_molecule).
        Expected kwds: nn_cutoff (float), degeneracy (1D array), state unused here.
        """

        return _compute_xtb_frame_from_fragments(
            xtb_model = self.xtb_model,
            fragments = fragments,
            nn_cutoff = nn_cutoff,
            degeneracy = degeneracy,
            state = state,
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
        elements = guess_types(u.atoms.names)
        u_worker.add_TopologyAttr("elements", elements)
        u_worker.load_new(traj_filename)
        u_worker.trajectory[frame_idx]
        fragments = u_worker.atoms.fragments
        return self.__call_local__(fragments, nn_cutoff, degeneracy, state)

def _atomgroup_to_xyz(
    atomgroup: AtomGroup,
) -> str:
    """Convert an MDAnalysis Atomgroup to an xyz

    Parameters
    ----------
    atomlist: AtomGroup
        MDAnalysis AtomGroup to convert

    Returns
    -------
    
    """
    # if isinstance(atomgroup, tuple): 
    #     atomgroup = sum(atomgroup[-1])

    # rdkit_mol = atomgroup.convert_to.rdkit()
    # xyz_block = rdmolfiles.MolToXYZBlock(rdkit_mol)
    elements = atomgroup.elements  # faster than atomgroup.names if elements available
    if len(atomgroup) == 2:
        positions = shift_dimer_images(atomgroup[0], atomgroup[1])
    else:
        positions = atomgroup.positions


    # Prepare formatted XYZ string
    lines = [f"{len(atomgroup)}"]
    lines += [f"{el} {x:.8f} {y:.8f} {z:.8f}"
              for el, (x, y, z) in zip(elements, positions)]

    xyz_block = "\n".join(lines)

    return xyz_block

def _convert_to_model_format(
    fragments: List[AtomGroup],
    nn_cutoff: float
    ) -> Dict[tuple,str]:
    """
    Find all dimer pairs within nn_cutoff, and convert each pair into a xyz block

    Parameters
    ----------

    Returns
    -------

    """

    for frag in fragments:
        mda.lib.mdamath.make_whole(frag)

    dimers = find_dimers(fragments, nn_cutoff)

    dimers_xyz: Dict[tuple, str] = {}
    logger.info('converting to xyz format')
    for (i, j), ag_pair in tqdm(dimers.items(), total=len(dimers.items()), desc='converting mdanalysis atoms to xyz'):
        mol = _atomgroup_to_xyz(ag_pair)
        dimers_xyz[(i,j)] = mol

    return dimers_xyz

def _compute_xtb_frame_from_fragments(
        fragments: List[AtomGroup],
        nn_cutoff: float,
        degeneracy: np.ndarray,
        xtb_model: Literal['gfn1-XTB', 'gfn2-XTB'] = 'gfn1-XTB',
        state: Literal['homo', 'lumo'] = 'homo',  
) -> np.ndarray:
    dimers_dict = _convert_to_model_format(fragments, nn_cutoff)
    size = degeneracy.sum()
    H_frag = np.zeros((size, size))
    stops = np.cumsum(degeneracy)
    starts = np.r_[0, stops[:-1]]
    diag = np.arange(size)
    wave = dict()  # in OcelotML scenario, we just store a dummy

    all_mols = list(dimers_dict.values())
    logger.info('running through dimer list')
    predictions = _xtb_from_list(
        dimers = all_mols,
        mode = state,
        flavour= xtb_model,
        )

    for idx, ((i, j), mol) in enumerate(dimers_dict.items()):
        ix, iy = starts[i], stops[i]
        jx, jy = starts[j], stops[j]

        H_frag[diag[ix:iy], diag[ix:iy]] = predictions[idx]  # coupling(i→j)
        H_frag[diag[jx:jy], diag[jx:jy]] = predictions[idx]  # symmetric

        wave[i] = 0
        wave[j] = 0

    for i in (set(range(len(degeneracy))) - set(wave.keys())):
        ix, iy = starts[i], stops[i]
        single_mol = _atomgroup_to_xyz(fragments[i])
        e_i = _xtb_from_list(
            molecule=single_mol,
            mode = state,
            flavour= xtb_model,
        )
        H_frag[diag[ix:iy], diag[ix:iy]] = e_i

    return H_frag, None

#OVERIDE THIS WHEN XTB COMPILES
XTB_EXECUTABLE = '/Users/k2584788/Downloads/xtb-bleed 2/build/xtb'

def _xtb_from_list(
    dimers: List[str],
    mode: Literal['homo', 'lumo'] = 'homo',
    flavour: Literal['gfn1-XTB', 'gfn2-XTB'] = 'gfn1-XTB',
    threshold: float = 0.1,
    xtb_executable: str = XTB_EXECUTABLE
) -> Tuple[float, ...]:
    """
    Run xTB DIPRO calculations on a list of dimer geometries (XYZ strings).

    Parameters
    ----------
        dimers: List of XYZ-format strings for each dimer.
        mode: 'homo' to extract hole-transport coupling, 'lumo' for electron-transport coupling.
        flavour: xTB flavor, either 'gfn1-XTB' or 'gfn2-XTB'.
        threshold: DIPRO energy-threshold in eV (used with --dipro).
        xtb_executable: Path or name of the xTB binary.

    Returns
    -------
        Tuple of coupling values (in eV) for each dimer in the same order.

    Raises
    ------
        XTBError: If xTB returns a non-zero exit code or parsing fails.
    """
    flavour_flags = {
        'gfn1-XTB': ['--gfn', '1'],
        'gfn2-XTB': ['--gfn', '2']
    }
    if flavour not in flavour_flags:
        raise ValueError(f"Unsupported xTB flavour: {flavour}")

    results: List[float] = []
    patterns = {
        'homo': re.compile(r"total \|J\(AB,eff\)\| for hole transport.*?:\s*([0-9.]+) eV", re.IGNORECASE),
        'lumo': re.compile(r"total \|J\(AB,eff\)\| for charge transport.*?:\s*([0-9.]+) eV", re.IGNORECASE),
    }
    pattern = patterns[mode]
    
    logger.info('running xtb DIPRO coupling across dimers')
    print('xtb run', flush=True)
    for idx, xyz_str in tqdm(enumerate(dimers), total = len(dimers)):
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False, mode='w') as tmp:
            tmp.write(xyz_str)
            tmp_filename = tmp.name

        cmd = [xtb_executable, tmp_filename, '--dipro', str(threshold)] + flavour_flags[flavour]
        logger.info(cmd)
        try:
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False
            )
        except FileNotFoundError as e:
            os.remove(tmp_filename)
            raise XTBError(f"xTB executable not found: {xtb_executable}") from e

        os.remove(tmp_filename)

        if completed.returncode != 0:
            print('failed molecule', xyz_str)

            raise XTBError(f"xTB failed for dimer index {idx}, exit code {completed.returncode}: {completed.stdout}")

        match = pattern.search(completed.stdout)
        if not match:
            raise XTBError(f"Failed to parse coupling for dimer index {idx}. Output:\n{completed.stdout}")

        coupling_value = float(match.group(1))
        results.append(coupling_value)

    return tuple(results)
    