from typing import Tuple


def parse_lmdb_key(key: str) -> Tuple[str, int]:
    """
    Parse a trajectory/frame key into (traj_name, frame_idx).

    Supports common patterns:
    - 'trajXYZ/000123'        -> ("trajXYZ", 123)
    - 'trajXYZ_frame_123'     -> ("trajXYZ", 123)

    Raises ValueError for unrecognized formats.
    """
    if not isinstance(key, str):
        try:
            key = key.decode('utf-8') 
        except Exception:
            raise ValueError(f"LMDB key must be str or decodable bytes, got {type(key)}")

    if '/' in key:
        t, f = key.split('/', 1)
        return t, int(f)

    if '_frame_' in key:
        t, f = key.rsplit('_frame_', 1)
        return t, int(f)

    raise ValueError(f"Unrecognized LMDB key format: {key}")


def extract_meta_from_value(value_bytes) -> Tuple[str, int]:
    """
    Decode a value payload and extract (traj_name, frame_idx) when metadata
    is stored inside the value (e.g., pickled dict with 'traj_name' and 'frame_idx').

    Implementers may adapt this to their serialization format.
    """
    import pickle, gzip
    try:
        obj = pickle.loads(gzip.decompress(value_bytes))
    except Exception:
        obj = pickle.loads(value_bytes)

    for t_key in ('traj_name', 'trajectory', 'traj'):
        if t_key in obj:
            traj = str(obj[t_key])
            break
    else:
        raise ValueError("Missing 'traj_name' in value payload")

    for f_key in ('frame_idx', 'frame', 'time_idx', 't'):
        if f_key in obj:
            frame_idx = int(obj[f_key])
            break
    else:
        fid = str(obj.get('id', ''))
        if 'frame_' in fid:
            try:
                frame_idx = int(fid.rsplit('frame_', 1)[1].split('.')[0])
            except Exception as e:
                raise ValueError(f"Could not infer frame index from id='{fid}': {e}")
        else:
            raise ValueError("Missing frame index in value payload")

    return traj, frame_idx

