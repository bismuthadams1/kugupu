from abc import ABC, abstractmethod
from MDAnalysis.core.groups import AtomGroup
import numpy as np
from typing import Any
from MDAnalysis import Universe
from typing import List, Type, Dict, Optional


MODELS_AVAILABLE: Dict[str, Type["CouplingModel"]]  = {}

class CouplingModel(ABC):

    _name: Optional[str] = None

    def __init__(self, *, local: bool = False, server_id: Optional[str] = None):
        """ Initialize the coupling model function

        Parameters
        ----------
        local : bool
            If True, calls go to __call_local__; otherwise to __call_remote__.
        server_id : str, optional
            Identifier (hostname, queue name, API key, etc.) to use for remote calls.
            Required only if local=False. If local=True, you may ignore server_id.
          
        """
        self.local = local
        self.server_id = server_id

    def __call__(self, *args, **kwds):
        """call the model"""
        if self.local:
            return self.__call_local__(*args, **kwds)
        else:
            return self.__call_remote__(*args, **kwds)

    def __init_subclass__(cls):
        super().__init_subclass__()

        if cls._name is None:
            raise ValueError(
                f"class {cls.__name__} must define a class attribute `_name`"
            )

        MODELS_AVAILABLE[cls._name] = cls

    @abstractmethod
    def __call_local__(self, fragments: List[AtomGroup], **kwds):
        """run a call to our coupling model locally"""
        ...

    @abstractmethod
    def __call_remote__(
        self,
        top_pickle: Any,       # e.g. u._topology
        traj_filename: str,    # e.g. u.trajectory.filename
        frame_idx: int,        # integer frame number
        nn_cutoff: float,
        degeneracy: np.ndarray,
        state: str,
    ) -> np.ndarray:
        """
        Compute the coupling matrix for a single frame remotely.
        Subclasses should rebuild the Universe on the worker via:
        u_worker = MDAnalysis.Universe(top_pickle)
        u_worker.load_new(traj_filename)
        u_worker.trajectory[frame_idx]
        then extract `u_worker.atoms.fragments` and compute H_frag.
        """
        ...