from numba.typed import Dict
from numba import types


def _make_numba_esri_dirmap():
    # ESRI direction mapping
    dirmap = {
        64: (-1, 0),  # North
        128: (-1, 1),  # Northeast
        1: (0, 1),  # East
        2: (1, 1),  # Southeast
        4: (1, 0),  # South
        8: (1, -1),  # Southwest
        16: (0, -1),  # West
        32: (-1, -1),  # Northwest
        -1: (0, 0),  # outlet/terminal
        -2: (0, 0),
        0: (0, 0),
    }

    # Create Numba-typed dictionary
    dirmap_numba = Dict.empty(
        key_type=types.int64, value_type=types.UniTuple(types.int64, 2)
    )

    for k, v in dirmap.items():
        dirmap_numba[k] = v

    return dirmap_numba
