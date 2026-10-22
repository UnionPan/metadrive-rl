"""Load NuRec reconstruction checkpoints without executing pickled code.

``checkpoint.ckpt`` inside a NuRec ``.usdz`` is a PyTorch Lightning checkpoint
whose pickle references a few NRE-internal classes (config enums). We load it
with ``weights_only=True`` and register inert stand-ins for those classes, so
only tensors and plain containers are reconstructed.
"""

from __future__ import annotations

import pickle
import re
import sys
import types
from typing import IO, Any

import torch

_UNSUPPORTED = re.compile(r"Unsupported global: GLOBAL (\S+) was not")


def _register_stub(qualified_name: str) -> None:
    module_name, _, class_name = qualified_name.rpartition(".")
    stub = type(
        class_name,
        (),
        {
            "__module__": module_name,
            "__init__": lambda self, *args, **kwargs: None,
            "__setstate__": lambda self, state: None,
        },
    )
    setattr(sys.modules.setdefault(module_name, types.ModuleType(module_name)), class_name, stub)
    torch.serialization.add_safe_globals([stub])


def load_state_dict(source: str | IO[bytes], max_stubs: int = 64) -> dict[str, Any]:
    for _ in range(max_stubs):
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            ckpt = torch.load(source, map_location="cpu", weights_only=True)
            return ckpt["state_dict"]
        except pickle.UnpicklingError as exc:
            match = _UNSUPPORTED.search(str(exc))
            if match is None:
                raise
            _register_stub(match.group(1))
    raise RuntimeError(f"checkpoint references more than {max_stubs} unknown globals")
