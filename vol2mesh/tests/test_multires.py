import os
import json
import tempfile
import unittest

import numpy as np

import faulthandler
faulthandler.enable()

from vol2mesh import Mesh
from vol2mesh.multires import (
    write_info,
    write_object_mesh,
    read_object_mesh,
    quantize_fragment_vertices,
    zorder_positions,
    trim_mesh_to_box,
    encode_fragment,
    encode_object_mesh,
    decode_fragment,
    build_info,
    split_mesh_into_cells,
    split_mesh_for_lod,
    encode_multilod_object,
    _octree_empty_placeholders,
)


def _cube_mesh(corner_xyz, size):
    """
    A simple closed cube (8 verts, 12 triangles) with its min corner at
    ``corner_xyz``, returned as (vertices_xyz, faces).
    """
    c = np.asarray(corner_xyz, dtype=np.float64)
    offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    vertices_xyz = c + size * offsets
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 6, 5], [4, 7, 6],  # top
        [0, 4, 5], [0, 5, 1],  # sides
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], dtype=np.uint32)
    return vertices_xyz, faces


class TestMultires(unittest.TestCase):

    def test_quantize_roundtrip(self):
        bits = 16
        max_q = (1 << bits) - 1
        chunk_shape = np.array([256.0, 256.0, 256.0])
        grid_origin = np.array([0.0, 0.0, 0.0])
        position = np.array([3, 5, 7])

        cell_corner = grid_origin + position * chunk_shape
        # Some fractional vertices well inside the cell.
        verts = cell_corner + np.array([
            [127.6, 12.3, 88.05],
            [0.0, 0.0, 0.0],
            [256.0, 256.0, 256.0],
            [200.123, 5.5, 255.9],
        ])

        q = quantize_fragment_vertices(verts, position, chunk_shape, grid_origin, 0, bits)
        assert q.dtype == np.uint16
        assert (q >= 0).all() and (q <= max_q).all()

        # Dequantize and check we're within one quantization step.
        decoded = cell_corner + chunk_shape * (q.astype(np.float64) / max_q)
        step = chunk_shape / max_q
        assert np.all(np.abs(decoded - verts) <= step + 1e-9)

        # Exact-corner verts must land on 0 and max_q.
        assert (q[1] == 0).all()
        assert (q[2] == max_q).all()

    def test_boundary_snap(self):
        # A vertex a hair past the far boundary (e.g. from a halo / smoothing)
        # should snap to max_q, not wrap or clip to something arbitrary.
        bits = 16
        max_q = (1 << bits) - 1
        chunk_shape = np.array([256.0, 256.0, 256.0])
        grid_origin = np.array([0.0, 0.0, 0.0])
        position = np.array([0, 0, 0])
        verts = np.array([[256.0001, -0.0001, 128.0]])
        q = quantize_fragment_vertices(verts, position, chunk_shape, grid_origin, 0, bits)
        assert q[0, 0] == max_q
        assert q[0, 1] == 0

    def test_zorder(self):
        # Z-curve order interleaves bits; for a 2x2x2 block the order is the
        # Morton sequence.
        positions = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ])
        # Shuffle, then check zorder restores Morton order.
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(positions))
        shuffled = positions[perm]
        order = zorder_positions(shuffled)
        ordered = shuffled[order]

        def morton(p):
            x, y, z = int(p[0]), int(p[1]), int(p[2])
            key = 0
            for b in range(2):
                key |= ((x >> b) & 1) << (3 * b + 0)
                key |= ((y >> b) & 1) << (3 * b + 1)
                key |= ((z >> b) & 1) << (3 * b + 2)
            return key

        keys = [morton(p) for p in ordered]
        assert keys == sorted(keys), f"Not in Morton order: {keys}"

    def test_write_and_read_object(self):
        d = tempfile.mkdtemp()
        bits = 16
        chunk = 256.0
        chunk_shape = np.array([chunk, chunk, chunk])
        grid_origin = np.array([0.0, 0.0, 0.0])
        segment_id = 12345

        # Build cubes in a few grid cells. Use Mesh objects (ZYX) for one and
        # raw (vertices_xyz, faces) for another, to exercise both inputs.
        cells = {
            (0, 0, 0): None,
            (2, 1, 3): None,
            (1, 0, 0): None,
        }
        expected = {}
        for cell in cells:
            corner = np.array(cell) * chunk
            v_xyz, faces = _cube_mesh(corner + 30.0, 100.0)
            expected[cell] = (v_xyz, faces)

        fragments = {
            (0, 0, 0): Mesh(expected[(0, 0, 0)][0][:, ::-1], expected[(0, 0, 0)][1]),
            (2, 1, 3): expected[(2, 1, 3)],   # raw (vertices_xyz, faces)
            (1, 0, 0): Mesh(expected[(1, 0, 0)][0][:, ::-1], expected[(1, 0, 0)][1]),
        }

        write_info(d, vertex_quantization_bits=bits,
                   transform=[8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0])
        n = write_object_mesh(d, segment_id, fragments, chunk_shape, grid_origin,
                              vertex_quantization_bits=bits)
        assert n == 3

        # info file sanity.
        with open(f"{d}/info") as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_multilod_draco"
        assert info["vertex_quantization_bits"] == bits
        assert len(info["transform"]) == 12

        assert os.path.exists(f"{d}/{segment_id}")
        assert os.path.exists(f"{d}/{segment_id}.index")

        result = read_object_mesh(d, segment_id)
        assert result["num_lods"] == 1
        assert result["num_fragments_per_lod"][0] == 3
        assert np.allclose(result["chunk_shape_xyz"], chunk_shape)

        # Match each decoded fragment back to its source cell and compare
        # vertices within one quantization step.
        step = chunk / ((1 << bits) - 1)
        assert len(result["fragments"]) == 3
        for frag in result["fragments"]:
            cell = tuple(int(c) for c in frag["position"])
            exp_v, exp_f = expected[cell]
            dec_v = frag["vertices_xyz"]
            assert len(dec_v) == len(exp_v)
            # Draco may reorder vertices; match by nearest.
            for ev in exp_v:
                dists = np.linalg.norm(dec_v - ev, axis=1)
                assert dists.min() <= np.sqrt(3) * step + 1e-6, \
                    f"vertex {ev} not recovered (min dist {dists.min()})"

    def test_trim_to_box(self):
        # A cube that overhangs its cell on the high-x side. Trimming should
        # cut the overhang at x=100 (introducing boundary vertices on that
        # plane) and drop everything beyond, leaving the mesh within the box.
        v_xyz, faces = _cube_mesh([50.0, 20.0, 20.0], 100.0)  # spans x in [50,150]
        box_lo = np.array([0.0, 0.0, 0.0])
        box_hi = np.array([100.0, 100.0, 100.0])

        tv, tf = trim_mesh_to_box(v_xyz, faces, box_lo, box_hi)
        assert len(tf) > 0
        # Every surviving vertex is inside the box (within fp tolerance).
        assert (tv >= box_lo - 1e-9).all() and (tv <= box_hi + 1e-9).all()
        # The cut created vertices lying exactly on the x=100 plane...
        assert np.isclose(tv[:, 0], 100.0).any()
        # ...whereas clipping would instead pile many vertices onto x=100 by
        # collapsing the overhang. Confirm trimming actually removed geometry
        # rather than retaining all original faces.
        assert len(tf) != len(faces)

    def test_encode_fragment_trim_vs_clip(self):
        # Same overhanging cube, encoded as a fragment with and without trim.
        chunk = np.array([100.0, 100.0, 100.0])
        origin = np.array([0.0, 0.0, 0.0])
        pos = (0, 0, 0)
        v_xyz, faces = _cube_mesh([50.0, 20.0, 20.0], 100.0)

        clipped = encode_fragment((v_xyz, faces), pos, chunk, origin, trim=False)
        trimmed = encode_fragment((v_xyz, faces), pos, chunk, origin, trim=True)
        assert clipped is not None and trimmed is not None

        import DracoPy
        v_clip = np.asarray(DracoPy.decode(clipped).points)
        v_trim = np.asarray(DracoPy.decode(trimmed).points)
        max_q = (1 << 16) - 1
        # Both keep vertices within the quantized cell range.
        for v in (v_clip, v_trim):
            assert (v >= 0).all() and (v <= max_q).all()
        # Trimming changes the geometry (cut + dropped overhang), so the
        # vertex set differs from the clip-only result.
        assert v_trim.shape != v_clip.shape or not np.array_equal(
            np.sort(v_trim, axis=0), np.sort(v_clip, axis=0))

    def test_build_info(self):
        info = build_info(vertex_quantization_bits=16,
                          transform=[8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0],
                          lod_scale_multiplier=2.0)
        assert info["@type"] == "neuroglancer_multilod_draco"
        assert info["vertex_quantization_bits"] == 16
        assert info["lod_scale_multiplier"] == 2.0
        assert len(info["transform"]) == 12

    def test_decode_fragment_inverts_encode(self):
        chunk = np.array([100.0, 100.0, 100.0])
        origin = np.array([0.0, 0.0, 0.0])
        pos = (1, 2, 3)
        v_xyz, faces = _cube_mesh(np.array(pos) * chunk + 20.0, 60.0)

        draco = encode_fragment((v_xyz, faces), pos, chunk, origin, vertex_quantization_bits=16)
        dv, df = decode_fragment(draco, pos, chunk, origin, vertex_quantization_bits=16)

        step = 100.0 / ((1 << 16) - 1)
        # Each original vertex is recovered to within ~one quantization step.
        for ev in v_xyz:
            assert np.linalg.norm(dv - ev, axis=1).min() <= np.sqrt(3) * step + 1e-6

    def test_encode_object_mesh_bytes_and_passthrough(self):
        chunk = np.array([256.0, 256.0, 256.0])
        origin = np.array([0.0, 0.0, 0.0])
        cells = {(0, 0, 0): None, (1, 0, 0): None, (2, 1, 3): None}
        for cell in cells:
            corner = np.array(cell) * 256.0
            cells[cell] = _cube_mesh(corner + 30.0, 100.0)  # (vertices_xyz, faces)

        # (a) Encoding to bytes and writing them out matches write_object_mesh on disk.
        data_bytes, index_bytes, n = encode_object_mesh(cells, chunk, origin)
        assert n == 3 and len(data_bytes) > 0 and len(index_bytes) > 0

        d = tempfile.mkdtemp()
        write_info(d, vertex_quantization_bits=16)
        with open(f"{d}/777", "wb") as f:
            f.write(data_bytes)
        with open(f"{d}/777.index", "wb") as f:
            f.write(index_bytes)
        res = read_object_mesh(d, 777)
        assert res["num_fragments_per_lod"][0] == 3

        # (b) Pre-encoded fragment bytes pass through verbatim and produce an
        # identical object to encoding the meshes directly.
        prebytes = {cell: encode_fragment(m, cell, chunk, origin) for cell, m in cells.items()}
        data2, index2, n2 = encode_object_mesh(prebytes, chunk, origin)
        assert (data2, index2, n2) == (data_bytes, index_bytes, n)

    def test_split_then_encode_roundtrip(self):
        # A mesh spanning multiple cells, split and encoded, decodes back in-cell.
        cell_size_zyx = np.array([64.0, 64.0, 64.0])
        N = 96
        zz, yy, xx = np.ogrid[:N, :N, :N]
        c = N / 2
        vol = ((zz - c)**2 + (yy - c)**2 + (xx - c)**2) <= 30**2
        mesh = Mesh.from_binary_vol(vol, method='skimage')

        frags = split_mesh_into_cells(mesh, cell_size_zyx)
        assert len(frags) > 1
        chunk_xyz = cell_size_zyx[::-1]
        data, index, n = encode_object_mesh(frags, chunk_xyz, [0, 0, 0])
        assert n == len(frags)

        d = tempfile.mkdtemp()
        write_info(d, vertex_quantization_bits=16)
        with open(f"{d}/1", "wb") as f:
            f.write(data)
        with open(f"{d}/1.index", "wb") as f:
            f.write(index)
        res = read_object_mesh(d, 1)
        ch = res['chunk_shape_xyz']
        for frag in res['fragments']:
            lo = res['grid_origin_xyz'] + frag['position'] * ch
            v = frag['vertices_xyz']
            assert (v >= lo - 1e-3).all() and (v <= lo + ch + 1e-3).all()

    def test_octree_empty_placeholders(self):
        # One occupied LOD-0 cell at (5,5,5); its ancestors must be filled.
        occupied = {0: {(5, 5, 5)}, 1: set(), 2: set()}
        empties = _octree_empty_placeholders(occupied, 3)
        assert empties[1] == {(2, 2, 2)}   # parent of (5,5,5)
        assert empties[2] == {(1, 1, 1)}   # grandparent
        # A coarse cell that is already occupied is not added as an empty.
        occupied2 = {0: {(2, 2, 2)}, 1: {(1, 1, 1)}}
        empties2 = _octree_empty_placeholders(occupied2, 2)
        assert empties2[1] == set()

    @staticmethod
    def _parse_positions_by_lod(index_bytes):
        buf = index_bytes
        pos = 0

        def take(dt, c):
            nonlocal pos
            a = np.frombuffer(buf, dtype=dt, count=c, offset=pos)
            pos += a.nbytes
            return a

        take("<f4", 3); take("<f4", 3)
        num_lods = int(take("<u4", 1)[0])
        take("<f4", num_lods); take("<f4", num_lods * 3)
        nfr = take("<u4", num_lods).astype(int)
        by_lod = {}
        for lod in range(num_lods):
            n = int(nfr[lod])
            p = take("<u4", n * 3).reshape(3, n).T
            take("<u4", n)  # offsets
            by_lod[lod] = {tuple(int(c) for c in row) for row in p}
        return num_lods, by_lod

    def test_multilod_roundtrip_partition_and_octree(self):
        # Build a 3-LOD object from a sphere, with small cells so each LOD has
        # several fragments and a real octree.
        N = 128
        zz, yy, xx = np.ogrid[:N, :N, :N]
        c = N / 2
        vol = ((zz - c)**2 + (yy - c)**2 + (xx - c)**2) <= 50**2
        mesh = Mesh.from_binary_vol(vol, method='skimage')

        chunk = np.array([32.0, 32.0, 32.0])
        num_lods = 3
        fragments_by_lod = {}
        current = mesh
        for lod in range(num_lods):
            if lod > 0:
                current = Mesh(current.vertices_zyx.copy(), current.faces.copy())
                current.simplify(0.5, preserve_border=True)
            fragments_by_lod[lod] = split_mesh_for_lod(current, chunk, lod)

        data, index, nfrags = encode_multilod_object(fragments_by_lod, chunk, [0, 0, 0])
        assert len(nfrags) == 3 and all(n > 0 for n in nfrags)

        # Octree ancestor-closure: every position at lod l has its parent at l+1.
        nl, pos_by_lod = self._parse_positions_by_lod(index)
        assert nl == 3
        for lod in range(num_lods - 1):
            for (x, y, z) in pos_by_lod[lod]:
                assert (x // 2, y // 2, z // 2) in pos_by_lod[lod + 1], \
                    f"missing parent of {(x, y, z)} at lod {lod + 1}"

        # Round-trip and validate per-fragment invariants.
        d = tempfile.mkdtemp()
        write_info(d, vertex_quantization_bits=16)
        with open(f"{d}/3", "wb") as fp:
            fp.write(data)
        with open(f"{d}/3.index", "wb") as fp:
            fp.write(index)
        res = read_object_mesh(d, 3)
        assert res['num_lods'] == 3

        max_q = (1 << 16) - 1
        for frag in res['fragments']:
            lod = frag['lod']
            position = frag['position']
            cell_size = chunk * (2 ** lod)
            lo = position * cell_size      # grid_origin == 0
            v = frag['vertices_xyz']
            faces = frag['faces']
            step = cell_size / max_q

            # In-cell.
            assert (v >= lo - 1e-3).all() and (v <= lo + cell_size + 1e-3).all()

            # 2x2x2 partition: for lod>0, no triangle may straddle a mid-plane.
            if lod > 0 and len(faces):
                mid = lo + cell_size / 2.0
                tri = v[faces]  # (T, 3, 3)
                for ax in range(3):
                    rel = tri[:, :, ax] - mid[ax]
                    below = (rel < -step[ax]).any(axis=1)
                    above = (rel > step[ax]).any(axis=1)
                    assert not (below & above).any(), \
                        f"triangle crosses mid-plane (axis {ax}, lod {lod})"

    def test_empty_object(self):
        d = tempfile.mkdtemp()
        chunk_shape = np.array([256.0, 256.0, 256.0])
        grid_origin = np.array([0.0, 0.0, 0.0])
        # Fragment with no faces -> skipped -> nothing written.
        fragments = {(0, 0, 0): (np.zeros((0, 3)), np.zeros((0, 3), np.uint32))}
        n = write_object_mesh(d, 999, fragments, chunk_shape, grid_origin)
        assert n == 0
        assert not os.path.exists(f"{d}/999")
        assert not os.path.exists(f"{d}/999.index")


if __name__ == "__main__":
    unittest.main()
