from typing import Dict, Tuple, Optional

_df_time = None
_coord_to_orig: Dict[Tuple[float, float], int] = {}

def set_time_df(df):
    global _df_time
    _df_time = df

def has_time() -> bool:
    return _df_time is not None

def get_time_by_orig(a_orig: int, b_orig: int) -> float:
    if _df_time is None:
        raise RuntimeError("Time matrix not set")
    return float(_df_time.loc[a_orig, b_orig])

def set_coord_map(coord_map: Dict[int, Tuple[float, float]]):
    global _coord_to_orig
    _coord_to_orig = {tuple(v): k for k, v in coord_map.items()}

def get_time_by_coord(a_coord: Tuple[float, float], b_coord: Tuple[float, float]) -> Optional[float]:
    if _df_time is None:
        return None
    a_key = tuple(a_coord)
    b_key = tuple(b_coord)
    if a_key in _coord_to_orig and b_key in _coord_to_orig:
        a_orig = _coord_to_orig[a_key]
        b_orig = _coord_to_orig[b_key]
        return float(_df_time.loc[a_orig, b_orig])
    return None
