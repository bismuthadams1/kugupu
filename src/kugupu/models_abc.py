from abc import ABC, abstractmethod
from MDAnalysis.core.groups import AtomGroup
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
        if not self.local:
            if server_id is None:
                raise ValueError("`server_id` must be provided when local=False")
            self.server_id = server_id
        else:
            self.server_id = None


    def __call__(self, fragments: List[AtomGroup], **kwds):
        """call the model"""
        if self.local:
            return self.__call_local__(fragments, **kwds)
        else:
            return self.__call_remote__(fragments, **kwds)

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
    def __call_remote__(self, fragments: List[AtomGroup], *args, **kwds):
        """send a call to our remote workers"""
        ...
    
    @abstractmethod
    def _convert_to_model_format(self, fragments: List[AtomGroup], **kwds):
        """convert to format required for submission to model"""