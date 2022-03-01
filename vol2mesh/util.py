from itertools import starmap
import numpy as np


def compute_nonzero_box(mask):
    """
    Given a mask image, return the bounding box
    of the nonzero voxels in the mask.
    
    Equivalent to:
    
        coords = np.transpose(np.nonzero(mask))
        if len(coords) == 0:
            return np.zeros((2, mask.ndim))
        box = np.array([coords.min(axis=0), 1+coords.max(axis=0)])
        return box
    
    ...but faster.
    
    Args:
        mask:
            A binary image
    
    Returns:
        box, e.g. [(1,2,3), (10, 20,30)]
        If the mask is completely empty, zeros are returned,
        e.g. [(0,0,0), (0,0,0)]
    """
    box = _compute_nonzero_box_numpy(mask)

    # If the volume is completely empty,
    # the helper returns an invalid box.
    # In that case, return zeros
    if (box[1] <= box[0]).any():
        return np.zeros_like(box)

    return box


def _compute_nonzero_box_numpy(mask):
    """
    Helper for compute_nonzero_box().
 
    Note:
        If the mask has no nonzero voxels, an "invalid" box is returned,
        i.e. the start is above the stop.
    """
    # start with an invalid box
    box = np.zeros((2, mask.ndim), np.int32)
    box[0, :] = mask.shape
    
    # For each axis, reduce along the other axes
    axes = [*range(mask.ndim)]
    for axis in axes:
        other_axes = tuple({*axes} - {axis})
        pos = np.logical_or.reduce(mask, axis=other_axes).nonzero()[0]
        if len(pos):
            box[0, axis] = pos[0]
            box[1, axis] = pos[-1]+1

    return box


def extract_subvol(array, box):
    """
    Extract a subarray according to the given box.
    """
    assert all(b >= 0 for b in box[0])
    assert all(b <= s for b,s in zip(box[1], array.shape))
    return array[box_to_slicing(*box)]


def box_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert bb_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )


def has_nonzero_edges(vol):
    """
    Return True if any of the voxels on the edge of the volume are non-zero.
    """
    nz = (
        vol[:, :, 0].any() or
        vol[:, :, -1].any() or

        vol[:, 0, :].any() or
        vol[:, -1, :].any() or

        vol[0, :, :].any() or
        vol[-1, :, :].any()
    )
    return nz
