from .models_abc import CouplingModel
from typing import List, Optional
from MDAnalysis.core.groups import AtomGroup
from openff.toolkit.topology import Molecule
from MDAnalysis.transformations.wrap import unwrap
import MDAnalysis as mda
from kugupu.dimers import find_dimers

class QCArchiveModel(CouplingModel):
    _name = "qcarchive"

    def __init__(self, *, local: bool = False, server_id: Optional["distributed.Client"] = None):
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
            self.universe = None 
            transformer = unwrap(self.universe.atoms, in_place=True)
            # we need to stop fragmentation of molecules across periodic boundaries
            self.transformed_universe = self.universe.trajectory.add_transformations(transformer)


    def __call_local__(
            self,
            fragments: List[str],  # Assuming fragments are represented as strings (e.g., SMILES)
            nn_cutoff: float,
            degeneracy: List[float],
            state: Optional[str] = 'homo'  # This is where we can pick between models
    ):
        pass

    def __call_remote__(self, top_pickle, traj_filename, frame_idx, nn_cutoff, degeneracy, state):



        return super().__call_remote__(top_pickle, traj_filename, frame_idx, nn_cutoff, degeneracy, state)
    
    def convert_to_charge_format():


    

def _prepare_qc_archive_input(
     fragments: List[AtomGroup],
     nn_cutoff: 
):
    for frag in fragments:
        mda.lib.mdamath.make_whole(frag)

    dimers = find_dimers(fragments, nn_cutoff)

def _conver_to_charge_format(
        atomgroup: AtomGroup
        ) -> Molecule:

    atomgroup.elements
    

            
    
# 1 convert AtomGroups to molecules - split into grouped and individual molecules
# 2 find method supplied
# 3 create unique dataset name based on simulation
# 4 ensure unique compute tag
# 5 submit calculations -  wavefunction return, perform coupling calculations
# 6 pull calculations when finished 
