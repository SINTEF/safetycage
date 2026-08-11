"""AutoModelModule: picks TorchModelModule or SklearnModelModule for you."""
from typing import Any

from safetycage.modelmodule import ModelModule
from safetycage.modelmodules.sklearn_modelmodule import SklearnModelModule
from safetycage.modelmodules.torch_modelmodule import HAS_TORCH, TorchModelModule

if HAS_TORCH:
    import torch


class AutoModelModule:
    """Factory that dispatches to TorchModelModule or SklearnModelModule.

    AutoModelModule(model, ``**kwargs``) returns a ModelModule instance
    directly; AutoModelModule is never itself instantiated.

    Any ``**kwargs`` not recognized by the dispatched class's constructor
    are silently absorbed by ``ModelModule.__init__``'s own ``**kwargs`` and
    have no effect — e.g. passing ``device=`` when the model dispatches to
    ``SklearnModelModule`` does nothing, no error.
    """

    def __new__(cls, model: Any, **kwargs: Any) -> ModelModule:
        if HAS_TORCH and isinstance(model, torch.nn.Module):
            return TorchModelModule(model, **kwargs)
        if hasattr(model, "predict"):
            return SklearnModelModule(model, **kwargs)
        raise TypeError(
            f"Could not determine a ModelModule for {type(model).__name__}: "
            "it is not a torch.nn.Module and has no predict() method. Write "
            "a custom ModelModule (see docs/how-it-works.md)."
        )
