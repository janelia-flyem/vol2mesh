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


def trim_mesh_to_box(vertices_xyz, faces, box_lo_xyz, box_hi_xyz):
    """
    Trim a mesh to an axis-aligned box by slicing its triangles at each of
    the box's 6 planes.  Triangles straddling a plane are cut (introducing
    new vertices exactly on the plane) and the outside portion is dropped;
    no caps are added, so the mesh stays "open" at the cut.

    This is the geometric alternative to coordinate clipping: instead of
    collapsing overhanging triangles onto the cell face, it removes the
    overhang and leaves clean boundary vertices on the face.  Adjacent
    fragments cut against the same shared plane therefore line up closely
    (especially if they were generated with a wide halo and smoothed
    identically).

    Requires the optional ``trimesh`` package.

    Args:
        vertices_xyz:
            (N, 3) float vertex array, XYZ order.
        faces:
            (M, 3) integer face array.
        box_lo_xyz, box_hi_xyz:
            (3,) float min/max corners of the box, XYZ order.

    Returns:
        (vertices_xyz, faces) of the trimmed mesh.
    """
    from trimesh.intersections import slice_faces_plane

    v = np.asarray(vertices_xyz, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    box_lo_xyz = np.asarray(box_lo_xyz, dtype=np.float64)
    box_hi_xyz = np.asarray(box_hi_xyz, dtype=np.float64)

    # (point on plane, inward normal). slice_faces_plane keeps the side the
    # normal points toward, so the inward normals retain the box interior.
    planes = [
        (box_lo_xyz, np.array([ 1.0,  0.0,  0.0])),
        (box_hi_xyz, np.array([-1.0,  0.0,  0.0])),
        (box_lo_xyz, np.array([ 0.0,  1.0,  0.0])),
        (box_hi_xyz, np.array([ 0.0, -1.0,  0.0])),
        (box_lo_xyz, np.array([ 0.0,  0.0,  1.0])),
        (box_hi_xyz, np.array([ 0.0,  0.0, -1.0])),
    ]
    for origin, normal in planes:
        if len(f) == 0:
            break
        v, f, _ = slice_faces_plane(v, f, normal, origin)

    return v, f


def encode_fragment(fragment_mesh, fragment_position,
                    chunk_shape_xyz, grid_origin_xyz,
                    lod=0, vertex_quantization_bits=16, trim=False):
    """
    Quantize and Draco-encode a single fragment.

    Args:
        fragment_mesh:
            A vol2mesh ``Mesh`` (ZYX vertices) or a ``(vertices_xyz, faces)``
            tuple.
        fragment_position, chunk_shape_xyz, grid_origin_xyz, lod,
        vertex_quantization_bits:
            See :func:`quantize_fragment_vertices`.
        trim:
            If True, geometrically trim the fragment to its grid cell (via
            :func:`trim_mesh_to_box`) before quantizing, cutting any triangles
            that overhang the cell.  This is preferable to the coordinate
            clipping that quantization falls back on, which merely flattens
            overhanging triangles onto the cell face.  Requires ``trimesh``.

    Returns:
        Draco-encoded ``bytes``, or ``None`` if the fragment has no
        geometry (no faces) and should be omitted.
    """
    vertices_xyz, faces = _as_vertices_xyz_faces(fragment_mesh)
    if len(vertices_xyz) == 0 or len(faces) == 0:
        return None

    if trim:
        cell_size = np.asarray(chunk_shape_xyz, dtype=np.float64) * (2 ** lod)
        cell_corner = (np.asarray(grid_origin_xyz, dtype=np.float64)
                       + np.asarray(fragment_position, dtype=np.int64) * cell_size)
        vertices_xyz, faces = trim_mesh_to_box(
            vertices_xyz, faces, cell_corner, cell_corner + cell_size)
        if len(faces) == 0:
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


def decode_fragment(draco_bytes, fragment_position, chunk_shape_xyz, grid_origin_xyz,
                    lod=0, vertex_quantization_bits=16, vertex_offset=None):
    """
    Decode a single Draco fragment and dequantize its integer vertex positions
    back to stored-model coordinates (the inverse of :func:`encode_fragment`).

    Args:
        draco_bytes:
            Draco-encoded fragment bytes (integer position attribute).
        fragment_position, chunk_shape_xyz, grid_origin_xyz, lod,
        vertex_quantization_bits:
            See :func:`quantize_fragment_vertices`.  Must match the values
            used to encode the fragment.
        vertex_offset:
            Optional (3,) per-LOD vertex offset (XYZ).  Defaults to zeros.

    Returns:
        ``(vertices_xyz, faces)`` -- float64 stored-model coordinates (XYZ)
        and uint32 faces.
    """
    mesh = DracoPy.decode(draco_bytes)
    q = np.asarray(mesh.points, dtype=np.float64)

    max_q = float((1 << vertex_quantization_bits) - 1)
    cell_size = np.asarray(chunk_shape_xyz, dtype=np.float64) * (2 ** lod)
    cell_corner = (np.asarray(grid_origin_xyz, dtype=np.float64)
                   + np.asarray(fragment_position, dtype=np.int64) * cell_size)
    if vertex_offset is None:
        vertex_offset = np.zeros(3, dtype=np.float64)

    vertices_xyz = cell_corner + vertex_offset + cell_size * (q / max_q)
    return vertices_xyz, np.asarray(mesh.faces, dtype=np.uint32)


def split_mesh_into_cells(mesh, cell_size_zyx):
    """
    Partition a mesh into per-grid-cell fragments.

    Each face is binned to the cell(s) its bounding box overlaps, and each
    cell's faces are then geometrically trimmed to the cell box (so a face
    straddling a cell boundary is cut and appears in every cell it touches,
    leaving no gaps).  The grid is origin-aligned, with cells of size
    ``cell_size_zyx``.

    Note: operates in the mesh's native ZYX coordinate space (matching
    ``Mesh.vertices_zyx``), but returns fragments keyed by XYZ grid-cell
    index, ready to pass to :func:`encode_object_mesh` / :func:`write_object_mesh`.

    Args:
        mesh:
            A vol2mesh ``Mesh``.
        cell_size_zyx:
            (3,) cell extents in the mesh's coordinate space, ZYX order.

    Returns:
        ``{(x, y, z): Mesh}`` keyed by integer grid-cell index, XYZ order.
    """
    from vol2mesh import Mesh

    v = np.asarray(mesh.vertices_zyx, dtype=np.float64)
    f = np.asarray(mesh.faces)
    if len(f) == 0:
        return {}

    cell_size_zyx = np.asarray(cell_size_zyx, dtype=np.float64)
    fv = v[f]                                                       # (F, 3, 3) zyx
    lo = np.floor(fv.min(axis=1) / cell_size_zyx).astype(np.int64)  # (F, 3) cell index
    hi = np.floor(fv.max(axis=1) / cell_size_zyx).astype(np.int64)

    # Build (cell_index, face_index) pairs. Single-cell faces (the common
    # case) are assigned in bulk; the few boundary-straddling faces are
    # enumerated over their (small) cell ranges.
    straddles = (lo != hi).any(axis=1)
    cell_rows = [lo[~straddles]]
    face_rows = [np.flatnonzero(~straddles)]
    for i in np.flatnonzero(straddles):
        for cz in range(lo[i, 0], hi[i, 0] + 1):
            for cy in range(lo[i, 1], hi[i, 1] + 1):
                for cx in range(lo[i, 2], hi[i, 2] + 1):
                    cell_rows.append(np.array([[cz, cy, cx]], dtype=np.int64))
                    face_rows.append(np.array([i]))
    cells = np.concatenate(cell_rows)                               # (P, 3)
    face_ids = np.concatenate(face_rows)                            # (P,)

    # Group face indices by cell.
    order = np.lexsort(cells.T)
    cells = cells[order]
    face_ids = face_ids[order]
    group_starts = np.r_[0,
                         1 + np.flatnonzero((cells[1:] != cells[:-1]).any(axis=1)),
                         len(cells)]

    fragments = {}
    for s, e in zip(group_starts[:-1], group_starts[1:]):
        cz, cy, cx = (int(c) for c in cells[s])
        sub_faces = f[face_ids[s:e]]
        used = np.unique(sub_faces)
        sub_v = v[used]
        sub_f = np.searchsorted(used, sub_faces)
        cell_lo = np.array([cz, cy, cx], dtype=np.float64) * cell_size_zyx
        tv, tf = trim_mesh_to_box(sub_v, sub_f, cell_lo, cell_lo + cell_size_zyx)
        if len(tf) == 0:
            continue
        fragments[(cx, cy, cz)] = Mesh(tv, tf)

    return fragments


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


def build_info(vertex_quantization_bits=16, transform=None,
               lod_scale_multiplier=1.0, segment_properties=None):
    """
    Construct the dataset-level ``info`` metadata as a dict.

    Args:
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

    Returns:
        The ``info`` dict (JSON-serializable).
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
    return info


def write_info(output_dir, vertex_quantization_bits=16,
               transform=None, lod_scale_multiplier=1.0,
               segment_properties=None):
    """
    Write the dataset-level ``info`` JSON file into ``output_dir``.
    See :func:`build_info` for the metadata fields.
    """
    info = build_info(vertex_quantization_bits, transform,
                      lod_scale_multiplier, segment_properties)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/info", "w") as f:
        json.dump(info, f)


def encode_object_mesh(fragments, chunk_shape_xyz, grid_origin_xyz,
                       vertex_quantization_bits=16, lod_scales=None, trim=False):
    """
    Encode a single object's fragments into the ``(data_bytes, index_bytes)``
    pair of the unsharded multires format (single level of detail), without
    touching the filesystem.  Use this when storing to a key/value store
    (e.g. DVID); use :func:`write_object_mesh` to write files.

    Args:
        fragments:
            A mapping ``{(x, y, z): item}`` from integer grid-cell position
            (XYZ order) to one of:
              - a vol2mesh ``Mesh``,
              - a ``(vertices_xyz, faces)`` tuple, or
              - already-encoded Draco ``bytes`` for that fragment, which are
                used verbatim (``trim`` and quantization are skipped -- the
                bytes must already have been produced with the same cell
                grid and quantization, e.g. by :func:`encode_fragment`).
            Empty fragments are skipped.
        chunk_shape_xyz, grid_origin_xyz, vertex_quantization_bits, trim:
            See :func:`encode_fragment`.
        lod_scales:
            Optional length-1 sequence with the LOD-0 scale value.  Defaults
            to ``[1.0]``.

    Returns:
        ``(data_bytes, index_bytes, num_fragments)``.  If the object has no
        geometry, returns ``(b'', b'', 0)``.
    """
    chunk_shape_xyz = np.asarray(chunk_shape_xyz, dtype=np.float64)
    grid_origin_xyz = np.asarray(grid_origin_xyz, dtype=np.float64)

    if lod_scales is None:
        lod_scales = [1.0]
    lod_scales = np.asarray(lod_scales, dtype=np.float64)
    if len(lod_scales) != 1:
        raise ValueError("This single-LOD encoder expects exactly one lod_scale")

    # Encode (or pass through) every non-empty fragment.
    positions = []
    encoded = []
    for position, item in fragments.items():
        if isinstance(item, (bytes, bytearray)):
            draco_bytes = bytes(item)
            if len(draco_bytes) <= 12:
                draco_bytes = None
        else:
            draco_bytes = encode_fragment(
                item, position, chunk_shape_xyz, grid_origin_xyz,
                lod=0, vertex_quantization_bits=vertex_quantization_bits, trim=trim)
        if draco_bytes is None:
            continue
        positions.append(np.asarray(position, dtype=np.int64))
        encoded.append(draco_bytes)

    if not encoded:
        return b'', b'', 0

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

    data_bytes = b''.join(encoded)

    index_bytes = b''.join([
        chunk_shape_xyz.astype("<f4").tobytes(),
        grid_origin_xyz.astype("<f4").tobytes(),
        struct.pack("<I", num_lods),
        lod_scales.astype("<f4").tobytes(),
        vertex_offsets.astype("<f4").tobytes(order="C"),
        num_fragments_per_lod.astype("<u4").tobytes(),
        # fragment_positions: C-order [3, num_fragments] => all x, all y, all z.
        positions.T.astype("<u4").tobytes(order="C"),
        # fragment_offsets: byte size of each fragment, in the same order.
        offsets.astype("<u4").tobytes(),
    ])

    return data_bytes, index_bytes, len(encoded)


def write_object_mesh(output_dir, segment_id, fragments,
                      chunk_shape_xyz, grid_origin_xyz,
                      vertex_quantization_bits=16, lod_scales=None, trim=False):
    """
    Write the ``<segment-id>`` data file and ``<segment-id>.index`` manifest
    for a single object (single LOD).  Thin filesystem wrapper around
    :func:`encode_object_mesh`; ``fragments`` accepts the same items.

    Returns:
        The number of fragments actually written (after skipping empties).
        Returns 0 without writing any files if the object has no geometry.
    """
    data_bytes, index_bytes, num_fragments = encode_object_mesh(
        fragments, chunk_shape_xyz, grid_origin_xyz,
        vertex_quantization_bits=vertex_quantization_bits,
        lod_scales=lod_scales, trim=trim)

    if num_fragments == 0:
        return 0

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{int(segment_id)}", "wb") as f:
        f.write(data_bytes)
    with open(f"{output_dir}/{int(segment_id)}.index", "wb") as f:
        f.write(index_bytes)

    return num_fragments


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

        for i in range(n):
            size = int(frag_offsets[i])
            chunk = data[data_pos:data_pos + size]
            data_pos += size
            if size == 0:
                continue
            position = frag_positions[i]
            vertices_xyz, faces = decode_fragment(
                chunk, position, chunk_shape, grid_origin,
                lod=lod, vertex_quantization_bits=vertex_quantization_bits,
                vertex_offset=vertex_offsets[lod])
            result["fragments"].append({
                "position": position,
                "lod": lod,
                "vertices_xyz": vertices_xyz,
                "faces": faces,
            })

    return result
