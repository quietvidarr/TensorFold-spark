"""Conservative image-aware namespace for FlashNext prefix snapshots.

Only the same complete processed-image history may reuse state. Appended text
is allowed; adding/replacing/reordering/resizing media is a cache miss. Videos
keep the upstream cold path. This is not a token-only image-placeholder key.
"""
import hashlib
import json
import numpy as np


def _array(h, value):
    array = np.ascontiguousarray(value)
    h.update(json.dumps([array.dtype.str, list(array.shape)], separators=(',', ':')).encode())
    h.update(memoryview(array).cast('B'))


def cache_key(prepared, prompt=None):
    if prepared is None:
        return None
    if any(not hasattr(prepared, name) for name in ('token_ids', 'image_spans', 'image_hashes',
            'position_ids', 'image_grid_thw', 'pixel_values', 'rope_delta')):
        return None
    if prompt is not None and tuple(prepared.token_ids) != tuple(prompt):
        raise ValueError('Vision preparation belongs to different prompt tokens')
    if getattr(prepared, 'video_spans', ()) or getattr(prepared, 'video_hashes', ()) or getattr(prepared, 'video_pixel_values', None) is not None:
        return None
    spans = tuple(prepared.image_spans)
    if not spans or len(spans) != len(prepared.image_hashes):
        return None
    if prepared.position_ids.shape != (3, 1, len(prepared.token_ids)) or prepared.image_grid_thw.shape != (len(spans), 3):
        raise ValueError('Invalid visual position/grid shape before cache selection')
    if prepared.pixel_values.ndim != 2 or any(int(v) != v or v <= 0 for v in prepared.image_grid_thw.flat):
        raise ValueError('Invalid visual pixels/grid before cache selection')
    if prepared.pixel_values.shape[0] != sum(int(np.prod(row)) for row in prepared.image_grid_thw):
        raise ValueError('Pixel rows do not match visual grids')
    end = max(stop for start, stop in spans)
    if end > len(prepared.token_ids) or any(start < 0 or stop <= start for start, stop in spans):
        raise ValueError('Invalid image spans for vision-prefix namespace')
    h = hashlib.sha256(b'flashnext-vision-prefix-v1\0')
    h.update(json.dumps([list(prepared.image_hashes), spans, int(prepared.rope_delta)], separators=(',', ':')).encode())
    _array(h, prepared.image_grid_thw)
    _array(h, prepared.pixel_values)
    _array(h, prepared.position_ids[..., :end])
    return h.hexdigest()


def reusable(prepared, key):
    return prepared is None or key is not None


def keepable(state):
    return state.image_positions is None or getattr(state, 'vision_cache_key', None) is not None


def snapshot_metadata(state):
    # Rotary positions are immutable per request. Reattachment replaces, never
    # mutates, this tensor. Features are NOT retained after prefill.
    return {'vision_cache_key': getattr(state, 'vision_cache_key', None),
            'image_positions': state.image_positions, 'rope_delta': state.rope_delta}
