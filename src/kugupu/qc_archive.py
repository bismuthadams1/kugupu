from .models_abc import CouplingModel
from typing import List, Optional


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