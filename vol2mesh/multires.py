"""
Functions to write the neuroglancer multi-resolution mesh format
(``"@type": "neuroglancer_multilod_draco"``).

The format is documented in the neuroglancer source tree at
``src/datasource/precomputed/meshes.md``.  A dataset consists of:

- an ``info`` JSON file describing the whole mesh layer,
- one ``<segment-id>.index`` binary manifest per object, and
- one ``<segment-id>`` binary data file per object, holding the
  concatenated Draco-encoded mesh fragments (one fragment per occupied
  grid cell, in Z-curve order).

This module implements the *unsharded* storage representation and is
currently aimed at the single-level-of-detail (single-LOD) case, which is
the common case when each object's mesh is assembled from independently
generated chunk meshes (one mesh per grid cell).  Multi-LOD output (octree
decomposition with the 2x2x2 partition requirement) is not implemented
here yet.

Coordinate spaces
-----------------
All geometry is expressed in a "stored model" coordinate space of your
choosing; the ``info`` file's ``transform`` maps that space to the global
"model" space (typically nanometers) that neuroglancer renders in.  A
natural choice is to use full-resolution voxel units for the stored model
space and put the voxel->nm scale in ``transform``.

Note that this format (and therefore this module) works in **XYZ** order,
whereas :class:`vol2mesh.mesh.Mesh` stores vertices in ZYX order.  The
fragment-encoding functions below accept ``Mesh`` objects (and flip to XYZ
internally) or raw ``(vertices_xyz, faces)`` tuples.  Grid parameters such
as ``chunk_shape_xyz`` and ``grid_origin_xyz`` must be given in XYZ order.

The per-fragment vertex quantization (continuous stored-model coordinates ->
integer lattice within a cell) and the boundary-snapping logic that keeps
adjacent fragments from disagreeing on shared boundary vertices were
adapted from the ``mesh-n-bone`` library's ``multires/decomposition.py``.
"""
import os
import json
import struct
from functools import cmp_to_key

import numpy as np

import DracoPy


def _as_vertices_xyz_faces(fragment_mesh):
    """
    Accept either a vol2mesh ``Mesh`` (vertices stored ZYX) or a raw
    ``(vertices_xyz, faces)`` pair, and return ``(vertices_xyz, faces)``
    with vertices in XYZ order as float64 and faces as uint32.
    """
    if hasattr(fragment_mesh, 'vertices_zyx'):
        vertices_xyz = np.asarray(fragment_mesh.vertices_zyx, dtype=np.float64)[:, ::-1]
        faces = np.asarray(fragment_mesh.faces, dtype=np.uint32)
    else:
        vertices_xyz, faces = fragment_mesh
        vertices_xyz = np.asarray(vertices_xyz, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.uint32)
    return vertices_xyz, faces


def quantize_fragment_vertices(vertices_xyz, fragment_position,
                               chunk_shape_xyz, grid_origin_xyz,
                               lod=0, vertex_quantization_bits=16):
    """
    Quantize continuous stored-model vertex coordinates to the integer
    lattice neuroglancer expects within a single fragment (grid cell).

    Each output coordinate is an integer ``q`` in ``[0, 2**bits)``.  On
    decode, neuroglancer recovers the stored-model coordinate as::

        grid_origin[j]
        + chunk_shape[j] * (2**lod) * (fragment_position[j] + q / (2**bits - 1))

    Vertices that fall (slightly) outside the cell are clipped to it, and
    vertices within half a quantization step of a cell boundary are snapped
    exactly onto it.  The snap matters because two adjacent fragments that
    share a boundary vertex must quantize it to the same lattice value;
    otherwise the boundary decodes to two slightly different world
    positions and a sub-pixel seam appears.

    Args:
        vertices_xyz:
            (N, 3) float array of stored-model coordinates, XYZ order.
        fragment_position:
            (3,) integer grid-cell index of this fragment, XYZ order.
        chunk_shape_xyz:
            (3,) float extents of a LOD-0 cell, XYZ order.
        grid_origin_xyz:
            (3,) float origin of the grid, XYZ order.
        lod:
            Level of detail of this fragment (0 for the finest).
        vertex_quantization_bits:
            Bits per coordinate; must be 10 or 16 per the spec.

    Returns:
        (N, 3) uint16 array of quantized integer coordinates.
    """
    if not (1 <= vertex_quantization_bits <= 16):
        raise ValueError("vertex_quantization_bits must be between 1 and 16")

    chunk_shape_xyz = np.asarray(chunk_shape_xyz, dtype=np.float64)
    grid_origin_xyz = np.asarray(grid_origin_xyz, dtype=np.float64)
    fragment_position = np.asarray(fragment_position, dtype=np.int64)

    cell_size = chunk_shape_xyz * (2 ** lod)
    cell_corner = grid_origin_xyz + fragment_position * cell_size

    max_q = float((1 << vertex_quantization_bits) - 1)

    local = np.asarray(vertices_xyz, dtype=np.float64) - cell_corner
    local = np.clip(local, 0.0, cell_size)

    # Snap vertices within half a quantization step of a boundary onto it,
    # so adjacent cells agree on shared boundary vertices.
    half_step = cell_size / max_q / 2.0
    local = np.where(local < half_step, 0.0, local)
    local = np.where(local > cell_size - half_step, cell_size, local)

    q = np.round(local * (max_q / cell_size))
    q = np.clip(q, 0, max_q).astype(np.uint16)
    return q


def encode_fragment(fragment_mesh, fragment_position,
                    chunk_shape_xyz, grid_origin_xyz,
                    lod=0, vertex_quantization_bits=16):
    """
    Quantize and Draco-encode a single fragment.

    Args:
        fragment_mesh:
            A vol2mesh ``Mesh`` (ZYX vertices) or a ``(vertices_xyz, faces)``
            tuple.
        fragment_position, chunk_shape_xyz, grid_origin_xyz, lod,
        vertex_quantization_bits:
            See :func:`quantize_fragment_vertices`.

    Returns:
        Draco-encoded ``bytes``, or ``None`` if the fragment has no
        geometry (no faces) and should be omitted.
    """
    vertices_xyz, faces = _as_vertices_xyz_faces(fragment_mesh)
    if len(vertices_xyz) == 0 or len(faces) == 0:
        return None

    q = quantize_fragment_vertices(
        vertices_xyz, fragment_position, chunk_shape_xyz,
        grid_origin_xyz, lod, vertex_quantization_bits)

    # Pass integer positions directly: neuroglancer's multilod_draco format
    # requires an integer position attribute and does NOT support Draco's
    # built-in quantization, so we must not pass quantization_* args here.
    draco_bytes = DracoPy.encode(points=q, faces=faces)

    # A valid Draco mesh header is larger than this; anything smaller means
    # all triangles were degenerate after quantization.
    if len(draco_bytes) <= 12:
        return None
    return draco_bytes


def _cmp_zorder(lhs, rhs):
    """
    Compare two (x, y, z) grid positions for Z-curve (Morton) order, using
    the 'most-significant differing bit' algorithm.

    This matches the canonical comparator used by igneous / cloud-volume
    (the de-facto neuroglancer multires producers): the last coordinate (z)
    is the most-significant axis, x the least-significant.  The tie-break
    direction is load-bearing for neuroglancer compatibility, so keep this
    in sync with that convention rather than "generalizing" it.
    """
    def less_msb(x, y):
        return x < y and x < (x ^ y)

    msd = len(lhs) - 1
    for dim in range(len(lhs) - 2, -1, -1):
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return int(lhs[msd]) - int(rhs[msd])


def zorder_positions(positions):
    """
    Return the indices that sort the given (N, 3) integer grid positions
    (XYZ order) into Z-curve order, as required for ``fragment_positions``.
    """
    order = sorted(range(len(positions)),
                   key=cmp_to_key(lambda a, b: _cmp_zorder(positions[a], positions[b])))
    return order


def write_info(output_dir, vertex_quantization_bits=16,
               transform=None, lod_scale_multiplier=1.0,
               segment_properties=None):
    """
    Write the dataset-level ``info`` JSON file.

    Args:
        output_dir:
            Directory to write the ``info`` file into (created if needed).
        vertex_quantization_bits:
            Must be 10 or 16 per the spec.
        transform:
            4x3 homogeneous transform (12 numbers, row-major) from the
            stored-model space to the model space (typically nm).  Defaults
            to the identity transform.
        lod_scale_multiplier:
            Factor applied to each ``lod_scales`` value from the manifests.
        segment_properties:
            Optional name of a segment-properties subdirectory.
    """
    if vertex_quantization_bits not in (10, 16):
        raise ValueError("vertex_quantization_bits must be 10 or 16")

    if transform is None:
        transform = [1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0]
    transform = list(np.asarray(transform, dtype=np.float64).reshape(-1))
    if len(transform) != 12:
        raise ValueError("transform must have 12 elements (4x3, row-major)")

    info = {
        "@type": "neuroglancer_multilod_draco",
        "vertex_quantization_bits": int(vertex_quantization_bits),
        "transform": transform,
        "lod_scale_multiplier": float(lod_scale_multiplier),
    }
    if segment_properties is not None:
        info["segment_properties"] = segment_properties

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/info", "w") as f:
        json.dump(info, f)


def write_object_mesh(output_dir, segment_id, fragments,
                      chunk_shape_xyz, grid_origin_xyz,
                      vertex_quantization_bits=16, lod_scales=None):
    """
    Write the ``<segment-id>`` data file and ``<segment-id>.index`` manifest
    for a single object, using a single level of detail (LOD 0).

    Args:
        output_dir:
            Directory to write into (created if needed).  This should be the
            same directory as the ``info`` file.
        segment_id:
            Integer object label.
        fragments:
            A mapping ``{(x, y, z): fragment_mesh}`` from integer grid-cell
            position (XYZ order) to a vol2mesh ``Mesh`` or a
            ``(vertices_xyz, faces)`` tuple.  Empty fragments are skipped.
        chunk_shape_xyz:
            (3,) float extents of a grid cell, XYZ order, in stored-model
            units.
        grid_origin_xyz:
            (3,) float origin of the grid, XYZ order, in stored-model units.
        vertex_quantization_bits:
            Must match the value in the ``info`` file (10 or 16).
        lod_scales:
            Optional length-1 sequence with the LOD-0 scale value.  Defaults
            to ``[1.0]``.

    Returns:
        The number of fragments actually written (after skipping empties).
        Returns 0 without writing any files if the object has no geometry.
    """
    chunk_shape_xyz = np.asarray(chunk_shape_xyz, dtype=np.float64)
    grid_origin_xyz = np.asarray(grid_origin_xyz, dtype=np.float64)

    if lod_scales is None:
        lod_scales = [1.0]
    lod_scales = np.asarray(lod_scales, dtype=np.float64)
    if len(lod_scales) != 1:
        raise ValueError("This single-LOD writer expects exactly one lod_scale")

    # Encode every non-empty fragment.
    positions = []
    encoded = []
    for position, fragment_mesh in fragments.items():
        draco_bytes = encode_fragment(
            fragment_mesh, position, chunk_shape_xyz, grid_origin_xyz,
            lod=0, vertex_quantization_bits=vertex_quantization_bits)
        if draco_bytes is None:
            continue
        positions.append(np.asarray(position, dtype=np.int64))
        encoded.append(draco_bytes)

    if not encoded:
        return 0

    positions = np.array(positions, dtype=np.int64)
    if (positions < 0).any():
        raise ValueError("fragment grid positions must be non-negative")

    # Fragments (and their byte offsets) must be stored in Z-curve order.
    order = zorder_positions(positions)
    positions = positions[order]
    encoded = [encoded[i] for i in order]
    offsets = np.array([len(b) for b in encoded], dtype=np.uint32)

    num_lods = 1
    vertex_offsets = np.zeros((num_lods, 3), dtype=np.float64)
    num_fragments_per_lod = np.array([len(encoded)], dtype=np.uint32)

    os.makedirs(output_dir, exist_ok=True)

    # Data file: concatenated Draco fragments in Z-curve order.
    data_path = f"{output_dir}/{int(segment_id)}"
    with open(data_path, "wb") as f:
        for b in encoded:
            f.write(b)

    # Manifest file.
    index_path = f"{output_dir}/{int(segment_id)}.index"
    with open(index_path, "wb") as f:
        f.write(chunk_shape_xyz.astype("<f4").tobytes())
        f.write(grid_origin_xyz.astype("<f4").tobytes())
        f.write(struct.pack("<I", num_lods))
        f.write(lod_scales.astype("<f4").tobytes())
        f.write(vertex_offsets.astype("<f4").tobytes(order="C"))
        f.write(num_fragments_per_lod.astype("<u4").tobytes())
        # fragment_positions: C-order [3, num_fragments] => all x, all y, all z.
        f.write(positions.T.astype("<u4").tobytes(order="C"))
        # fragment_offsets: byte size of each fragment, in the same order.
        f.write(offsets.astype("<u4").tobytes())

    return len(encoded)


def read_object_mesh(output_dir, segment_id, vertex_quantization_bits=None):
    """
    Parse a single object's ``.index`` manifest and data file and decode its
    fragments.  Intended for testing and inspection (it mirrors what
    neuroglancer does on the client side).

    ``vertex_quantization_bits`` is needed to dequantize the integer vertex
    positions; it lives in the dataset ``info`` file, not the manifest.  If
    not given, it is read from ``{output_dir}/info`` (falling back to 16).

    Returns a dict with the parsed manifest fields plus a ``"fragments"``
    list.  Each fragment entry is a dict with:

        - ``"position"``: (3,) int grid-cell position, XYZ.
        - ``"lod"``: level of detail.
        - ``"vertices_xyz"``: (N, 3) float64 dequantized stored-model
          coordinates, XYZ.
        - ``"faces"``: (M, 3) uint32.
    """
    if vertex_quantization_bits is None:
        try:
            with open(f"{output_dir}/info") as f:
                vertex_quantization_bits = int(json.load(f)["vertex_quantization_bits"])
        except (FileNotFoundError, KeyError):
            vertex_quantization_bits = 16

    index_path = f"{output_dir}/{int(segment_id)}.index"
    data_path = f"{output_dir}/{int(segment_id)}"

    with open(index_path, "rb") as f:
        buf = f.read()

    pos = 0

    def take(dtype, count):
        nonlocal pos
        arr = np.frombuffer(buf, dtype=dtype, count=count, offset=pos)
        pos += arr.nbytes
        return arr

    chunk_shape = take("<f4", 3).astype(np.float64)
    grid_origin = take("<f4", 3).astype(np.float64)
    num_lods = int(take("<u4", 1)[0])
    lod_scales = take("<f4", num_lods).astype(np.float64)
    vertex_offsets = take("<f4", num_lods * 3).astype(np.float64).reshape(num_lods, 3)
    num_fragments_per_lod = take("<u4", num_lods).astype(np.int64)

    result = {
        "chunk_shape_xyz": chunk_shape,
        "grid_origin_xyz": grid_origin,
        "num_lods": num_lods,
        "lod_scales": lod_scales,
        "vertex_offsets": vertex_offsets,
        "num_fragments_per_lod": num_fragments_per_lod,
        "fragments": [],
    }

    with open(data_path, "rb") as f:
        data = f.read()

    data_pos = 0
    for lod in range(num_lods):
        n = int(num_fragments_per_lod[lod])
        # fragment_positions: C-order [3, n] => reshape then transpose back to (n, 3).
        frag_positions = take("<u4", n * 3).reshape(3, n).T.astype(np.int64)
        frag_offsets = take("<u4", n).astype(np.int64)

        max_q = float((1 << vertex_quantization_bits) - 1)
        for i in range(n):
            size = int(frag_offsets[i])
            chunk = data[data_pos:data_pos + size]
            data_pos += size
            if size == 0:
                continue
            mesh = DracoPy.decode(chunk)
            q = np.asarray(mesh.points, dtype=np.float64)
            position = frag_positions[i]
            cell_size = chunk_shape * (2 ** lod)
            cell_corner = grid_origin + position * cell_size
            vertices_xyz = (cell_corner
                            + vertex_offsets[lod]
                            + cell_size * (q / max_q))
            result["fragments"].append({
                "position": position,
                "lod": lod,
                "vertices_xyz": vertices_xyz,
                "faces": np.asarray(mesh.faces, dtype=np.uint32),
            })

    return result
