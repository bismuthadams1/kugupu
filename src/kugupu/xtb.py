from models_abc import CouplingModel


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
