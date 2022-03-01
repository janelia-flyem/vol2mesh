import os
import sys
import glob
import logging
import tarfile
import functools
import subprocess
from io import BytesIO
from itertools import chain

import numpy as np
import lz4.frame
from vol2mesh.util import compute_nonzero_box, extract_subvol, has_nonzero_edges

try:
    from dvidutils import encode_faces_to_drc_bytes, decode_drc_bytes_to_faces
    _dvidutils_available = True
except ImportError:
    _dvidutils_available = False

from .normals import compute_face_normals, compute_vertex_normals
from .obj_utils import write_obj, read_obj
from .ngmesh import read_ngmesh, write_ngmesh
from .io_utils import TemporaryNamedPipe, AutoDeleteDir, stdout_redirected

logger = logging.getLogger(__name__)

DRACO_USE_PIPE = False

class Mesh:
    """
    A class to hold the elements of a mesh.
    """
    MESH_FORMATS = ('obj', 'drc', 'ngmesh')
    
    def __init__(self, vertices_zyx, faces, normals_zyx=None, box=None, pickle_compression_method='lz4'):
        """
        Args:
            vertices_zyx: ndarray (N,3), float32
            
            faces: ndarray (M,3), integer
                Each element is an index referring to an element of vertices_zyx
        
            normals_zyx: ndarray (N,3), float
            
            box: ndarray (2,3)
                Overall bounding box of the mesh.
                (The bounding box information is not stored in mesh files like .obj and .drc,
                but it is useful to store it here for programmatic manipulation.)
            
            pickle_compression_method:
                How (or whether) to compress vertices, normals, and faces during pickling.
                Choices are: 'draco', 'lz4', or None.
        """
        assert pickle_compression_method in (None, 'lz4', 'draco')
        self.pickle_compression_method = pickle_compression_method
        self._destroyed = False
        
        # Note: When restoring from pickled data, vertices and faces
        #       are restored lazily, upon first access.
        #       See __getstate__().
        self._vertices_zyx = np.asarray(vertices_zyx, dtype=np.float32)
        self._faces = np.asarray(faces, dtype=np.uint32)
        self._draco_bytes = None
        self._lz4_items = None

        if normals_zyx is None:
            self._normals_zyx = np.zeros((0,3), dtype=np.int32)
        else:
            self._normals_zyx = np.asarray(normals_zyx, np.float32)
            assert self._normals_zyx.shape in (self.vertices_zyx.shape, (0,3)), \
                "Normals were provided, but they don't match the shape of the vertices:\n" \
                f" {self._normals_zyx.shape} != {self.vertices_zyx.shape}"

        for a in (self._vertices_zyx, self._faces, self._normals_zyx):
            assert a.ndim == 2 and a.shape[1] == 3, f"Input array has wrong shape: {a.shape}"

        if box is not None:
            self.box = np.asarray(box)
            assert self.box.shape == (2,3) 
        elif len(self.vertices_zyx) == 0:
                # No vertices. Choose a box with huge "negative shape",
                # so that it will have no effect when merged with other meshes.
                MAX_INT = np.iinfo(np.int32).max
                MIN_INT = np.iinfo(np.int32).min
                self.box = np.array([[MAX_INT, MAX_INT, MAX_INT],
                                     [MIN_INT, MIN_INT, MIN_INT]], dtype=np.int32)
        else:
            self.box = np.array( [ self.vertices_zyx.min(axis=0),
                                   np.ceil( self.vertices_zyx.max(axis=0) ) ] ).astype(np.int32)


    def uncompressed_size(self):
        """
        Return the size of the uncompressed mesh data in bytes
        """
        return self.vertices_zyx.nbytes + self.normals_zyx.nbytes + self.faces.nbytes


    @classmethod
    def from_file(cls, path):
        """
        Alternate constructor.
        Read a mesh from .obj or .drc
        """
        ext = os.path.splitext(path)[1]
        
        # By special convention,
        # we permit 0-sized files, which result in empty meshes
        if os.path.getsize(path) == 0:
            return Mesh(np.zeros((0,3), np.float32),
                        np.zeros((0,3), np.uint32))
        
        if ext == '.drc':
            with open(path, 'rb') as drc_stream:
                draco_bytes = drc_stream.read()
            return Mesh.from_buffer(draco_bytes, 'drc')

        elif ext == '.obj':
            with open(path, 'rb') as obj_stream:
                vertices_xyz, faces, normals_xyz = read_obj(obj_stream)
                vertices_zyx = vertices_xyz[:,::-1]
                normals_zyx = normals_xyz[:,::-1]
            return Mesh(vertices_zyx, faces, normals_zyx)
        elif ext == '.ngmesh':
            with open(path, 'rb') as ngmesh_stream:
                vertices_xyz, faces = read_ngmesh(ngmesh_stream)
            return Mesh(vertices_xyz[:,::-1], faces)
        else:
            msg = f"Unknown file type: {path}"
            logger.error(msg)
            raise RuntimeError(msg)


    @classmethod
    def from_directory(cls, path, keep_normals=True):
        """
        Alternate constructor.
        Read all mesh files (either .drc or .obj) from a
        directory and concatenate them into one big mesh.
        """
        mesh_paths = chain(*[glob.glob(f'{path}/*.{ext}') for ext in cls.MESH_FORMATS])
        mesh_paths = sorted(mesh_paths)
        meshes = map(Mesh.from_file, mesh_paths)
        return concatenate_meshes(meshes, keep_normals)


    @classmethod
    def from_tarfile(cls, path_or_bytes, keep_normals=True, concatenate=True):
        """
        Alternate constructor.
        Read all mesh files (either .drc or .obj) from a .tar file
        and concatenate them into one big mesh, or return them as a dict of
        ``{name : mesh}`` items.
        
        Args:
            path_or_bytes:
                Either a path to a .tar file, or a bytes object
                containing the contents of a .tar file.
            
            keep_normals:
                Whether to keep the normals in the given meshes or discard them.
                If not all of the meshes in the tarfile contain normals,
                you will need to discard them.
            
            concatenate:
                If True, concatenate all meshes into a single ``Mesh`` object.
                Otherwise, return a dict of ``{name : Mesh}`` items,
                named according to the names in the tarfile.
        
        Note: The tar file structure should be completely flat,
        i.e. no internal directory.
        
        Returns:
            Either a single ``Mesh``, or a dict of ``{name : Mesh}``,
            depending on ``concatenate``.
        """
        if isinstance(path_or_bytes, str):
            tf = tarfile.open(path_or_bytes)
        else:
            tf = tarfile.TarFile(fileobj=BytesIO(path_or_bytes))
        
        # As a convenience, we sort the members by name before loading them.
        # This ensures that tarball storage order doesn't affect vertex order.
        members = sorted(tf.getmembers(), key=lambda m: m.name)

        meshes = {}
        for member in members:
            ext = os.path.splitext(member.name)[1][1:]
            
            # Skip non-mesh files and empty files            
            if ext in cls.MESH_FORMATS and member.size > 0:
                buf = tf.extractfile(member).read()
                try:
                    mesh = Mesh.from_buffer(buf, ext)
                except:
                    logger.error(f"Could not decode {member.name} ({member.size} bytes). Skipping!")
                    continue

                meshes[member.name] = mesh

        if concatenate:
            return concatenate_meshes(meshes.values(), keep_normals)
        else:
            return meshes


    @classmethod
    def from_buffer(cls, serialized_bytes, fmt):
        """
        Alternate constructor.
        Read a mesh from either .obj or .drc format, from a buffer.
        
        Args:
            serialized_bytes:
                bytes object containing the .obj or .drc file contents
            fmt:
                Either 'obj' or 'drc'.
        """
        assert fmt in cls.MESH_FORMATS
        if len(serialized_bytes) == 0:
            return Mesh(np.zeros((0,3), np.float32), np.zeros((0,3), np.uint32))

        if fmt == 'obj':
            with BytesIO(serialized_bytes) as obj_stream:
                vertices_xyz, faces, normals_xyz = read_obj(obj_stream)
                vertices_zyx = vertices_xyz[:,::-1]
                normals_zyx = normals_xyz[:,::-1]
            return Mesh(vertices_zyx, faces, normals_zyx)

        elif fmt == 'drc':
            assert _dvidutils_available, \
                "Can't read draco meshes if dvidutils isn't installed"

            vertices_xyz, normals_xyz, faces = decode_drc_bytes_to_faces(serialized_bytes)
            vertices_zyx = vertices_xyz[:,::-1]
            normals_zyx = normals_xyz[:,::-1]
            return Mesh(vertices_zyx, faces, normals_zyx)

        elif fmt == 'ngmesh':
            with BytesIO(serialized_bytes) as ngmesh_stream:
                vertices_xyz, faces = read_ngmesh(ngmesh_stream)
            return Mesh(vertices_xyz[:,::-1], faces)


    @classmethod
    def from_binary_vol(cls, downsampled_volume_zyx, fullres_box_zyx=None, method='ilastik', **kwargs):
        """
        Alternate constructor.
        Run marching cubes on the given volume and return a Mesh object.
        
        Args:
            downsampled_volume_zyx:
                A binary volume, possibly at a downsampled resolution.
            fullres_box_zyx:
                The bounding-box inhabited by the given volume, in FULL-res coordinates.
            method:
                Which library to use for marching_cubes. Choices are:
                - "ilastik" -- Use github.com/ilastik/marching_cubes
                - "skimage" -- Use scikit-image marching_cubes_lewiner
                  (Not a required dependency.  Install ``scikit-image`` to use this method.)
            kwargs:
                Any extra arguments to the particular marching cubes implementation.
                The 'ilastik' method supports initial smoothing via a ``smoothing_rounds`` parameter.
        
        Returns:
            Mesh

        Note:
            No surface is added for the volume boundaries, so objects which
            touch the edge of the volume will be "open" at the edge.
            If you want to see an edge there, pad your volume with a 1-px
            halo on all sides (and adjust fullres_box_zyx accordingly).
        """
        assert downsampled_volume_zyx.ndim == 3
        
        if fullres_box_zyx is None:
            fullres_box_zyx = np.array([(0,0,0), downsampled_volume_zyx.shape])
        else:
            fullres_box_zyx = np.asarray(fullres_box_zyx)
        
        # Infer the resolution of the downsampled volume
        resolution = (fullres_box_zyx[1] - fullres_box_zyx[0]) // downsampled_volume_zyx.shape

        try:
            if method == 'skimage':
                from skimage.measure import marching_cubes
                padding = np.array([0,0,0])
                
                # Tiny volumes trigger a corner case in skimage, so we pad them with zeros.
                # This results in faces on all sides of the volume,
                # but it's not clear what else to do.
                if (np.array(downsampled_volume_zyx.shape) <= 2).any():
                    padding = np.array([2,2,2], dtype=int) - downsampled_volume_zyx.shape
                    padding = np.maximum([0,0,0], padding)
                    downsampled_volume_zyx = np.pad( downsampled_volume_zyx, tuple(zip(padding, padding)), 'constant' )

                kws = {'step_size': 1}
                kws.update(kwargs)
                vertices_zyx, faces, normals_zyx, _values = marching_cubes(downsampled_volume_zyx, 0.5, **kws)
                
                # Skimage assumes that the coordinate origin is CENTERED inside pixel (0,0,0),
                # whereas we assume that the origin is the UPPER-LEFT corner of pixel (0,0,0).
                # Therefore, shift the results by a half-pixel.
                vertices_zyx += 0.5

                if padding.any():
                    vertices_zyx -= padding
            elif method == 'ilastik':
                from marching_cubes import march
                try:
                    smoothing_rounds = kwargs['smoothing_rounds']
                except KeyError:
                    smoothing_rounds = 0

                # ilastik's marching_cubes expects FORTRAN order
                if downsampled_volume_zyx.flags['F_CONTIGUOUS']:
                    vertices_zyx, normals_zyx, faces = march(downsampled_volume_zyx, smoothing_rounds)
                else:
                    downsampled_volume_zyx = np.asarray(downsampled_volume_zyx, order='C')
                    vertices_xyz, normals_xyz, faces = march(downsampled_volume_zyx.transpose(), smoothing_rounds)
                    vertices_zyx = vertices_xyz[:, ::-1]
                    normals_zyx = normals_xyz[:, ::-1]
                    faces[:] = faces[:, ::-1]

                vertices_zyx += 0.5
                
            else:
                msg = f"Unknown method: {method}"
                logger.error(msg)
                raise RuntimeError(msg)
        except ValueError:
            if downsampled_volume_zyx.all() or not downsampled_volume_zyx.any():
                # Completely full (or empty) boxes are not meshable -- they would be
                # open on all sides, leaving no vertices or faces.
                # Just return an empty mesh.
                empty_vertices = np.zeros( (0, 3), dtype=np.float32 )
                empty_faces = np.zeros( (0, 3), dtype=np.uint32 )
                return Mesh(empty_vertices, empty_faces, box=fullres_box_zyx)
            else:
                logger.error("Error during mesh generation")
                raise
    
        
        # Upscale and translate the mesh into place
        vertices_zyx[:] *= resolution
        vertices_zyx[:] += fullres_box_zyx[0]
        
        return Mesh(vertices_zyx, faces, normals_zyx, fullres_box_zyx)


    @classmethod
    def from_label_volume(cls, downsampled_volume_zyx, fullres_box_zyx=None, labels=None, ensure_halo=True, method='ilastik', progress=True, **kwargs):
        """
        Generate a mesh for multiple labels in a segmentation volume.
        Calls ``Mesh.from_binary_volume()`` for each object.

        Args:
            downsampled_volume_zyx:
                A label (segmentation) volume, possibly at a downsampled resolution.
            fullres_box_zyx:
                The bounding-box inhabited by the given volume, in FULL-res coordinates.
            labels:
                If given only compute meshes for the given labels in the volume.
                If any of the given labels cannot be found in the volume,
                ``None`` is returned in place of mesh object for that label.
                If no labels are provided, all non-zero labels are processed.
            ensure_halo:
                Ensure that the volume has an empty slice of voxels on all sides.
                Otherwise, meshes which border the volume may have 'holes' at the volume edge.
            method:
                Which library to use for marching_cubes. Choices are:
                - "ilastik" -- Use github.com/ilastik/marching_cubes
                - "skimage" -- Use scikit-image marching_cubes_lewiner
                  (Not a required dependency.  Install ``scikit-image`` to use this method.)
            progress:
                Show a progress bar if tqdm is installed.
            kwargs:
                Any extra arguments to the particular marching cubes implementation.
                The 'ilastik' method supports initial smoothing via a ``smoothing_rounds`` parameter.

        Returns:
            dict of ``{label: Mesh}``
        """
        if fullres_box_zyx is None:
            fullres_box_zyx = np.array([[0,0,0], downsampled_volume_zyx.shape])
        fullres_shape = fullres_box_zyx[1] - fullres_box_zyx[0]
        resolution = fullres_shape // downsampled_volume_zyx.shape

        # The fullres start/end do not need to be even multiples of the resolution,
        # but the *width* of each dimension must divide cleanly.
        assert not (fullres_shape % downsampled_volume_zyx.shape).any(), \
            "Mask volume dimensions must divide cleanly into full-res dimensions."

        if ensure_halo and has_nonzero_edges(downsampled_volume_zyx):
            downsampled_volume_zyx = np.pad(downsampled_volume_zyx, 1)
            fullres_box_zyx += resolution * np.array([[-1, -1, -1], [1, 1, 1]])

        if labels is None:
            # Which labels are present?
            # (Use pandas if available, since it's faster.)
            try:
                import pandas as pd
                labels = pd.unique(downsampled_volume_zyx.reshape(-1))
            except ImportError:
                labels = np.unique(downsampled_volume_zyx)

            labels = sorted({*labels} - {0})

        if progress:
            try:
                from tqdm import tqdm
                labels = tqdm(labels)
            except ImportError:
                pass

        meshes = {}
        for label in labels:
            mask = (downsampled_volume_zyx == label)

            # Save time by extracting the smallest
            # bounding box possible for the object.
            subvol_box = compute_nonzero_box(mask)
            if not subvol_box.any():
                meshes[label] = None
                continue

            subvol_box[0] = np.maximum(0, subvol_box[0] - 1)
            subvol_box[1] = np.minimum(mask.shape, subvol_box[1] + 1)

            subvol_mask = extract_subvol(mask, subvol_box)
            mesh = cls.from_binary_vol(subvol_mask, subvol_box, method, **kwargs)

            # Upscale and translate the mesh into place
            mesh.vertices_zyx[:] *= resolution
            mesh.vertices_zyx[:] += fullres_box_zyx[0]
            meshes[label] = mesh

        return meshes


    @classmethod
    def from_binary_blocks(cls, downsampled_binary_blocks, fullres_boxes_zyx, stitch=True, method='skimage'):
        """
        Alternate constructor.
        Compute a mesh for each of the given binary volumes
        (scaled and translated according to its associated box),
        and concatenate them (but not stitch them).
        
        Args:
            downsampled_binary_blocks:
                List of binary blocks on which to run marching cubes.
                The blocks need not be full-scale; their meshes will be re-scaled
                according to their corresponding bounding-boxes in fullres_boxes_zyx.

            fullres_boxes_zyx:
                List of bounding boxes corresponding to the blocks.
                Each block meshes will be re-scaled to fit exactly within it's bounding box.
            
            stitch:
                If True, deduplicate the vertices in the final mesh and topologically
                connect the faces in adjacent blocks.
            
            method:
                Which library to use for marching_cubes. Currently, only 'skimage' is supported.
        """
        meshes = []
        for binary_vol, fullres_box_zyx in zip(downsampled_binary_blocks, fullres_boxes_zyx):
            mesh = cls.from_binary_vol(binary_vol, fullres_box_zyx, method)
            meshes.append(mesh)

        mesh = concatenate_meshes(meshes)
        if stitch:
            mesh.stitch_adjacent_faces()
        return mesh


    def drop_normals(self):
        """
        Drop normals from the mesh.
        """
        self.normals_zyx = np.zeros((0,3), np.float32)


    def compress(self, method='lz4'):
        """
        Compress the array members of this mesh, and return the (approximate) compressed size.
        
        Method 'lz4' preserves data without loss.
        Method 'draco' is lossy.
        Method None will not compress at all.
        """
        if method is None:
            return self.vertices_zyx.nbytes + self.faces.nbytes + self.normals_zyx.nbytes
        elif method == 'draco':
            return self._compress_as_draco()
        elif method == 'lz4':
            return self._compress_as_lz4()
        else:
            raise RuntimeError(f"Unknown compression method: {method}")
    

    def _compress_as_draco(self):
        assert _dvidutils_available, \
            "Can't use draco compression if dvidutils isn't installed"
        if self._draco_bytes is None:
            self._uncompress() # Ensure not currently compressed as lz4
            self._draco_bytes = encode_faces_to_drc_bytes(self._vertices_zyx[:,::-1], self._normals_zyx[:,::-1], self._faces)
            self._vertices_zyx = None
            self._normals_zyx = None
            self._faces = None
        return len(self._draco_bytes)
    

    def _compress_as_lz4(self):
        if self._lz4_items is None:
            self._uncompress() # Ensure not currently compressed as draco
            compressed = []
            
            flat_vertices = self._vertices_zyx.reshape(-1)
            compressed.append( lz4.frame.compress(flat_vertices) )
            self._vertices_zyx = None
            
            flat_normals = self._normals_zyx.reshape(-1)
            compressed.append( lz4.frame.compress(flat_normals) )
            self._normals_zyx = None
    
            flat_faces = self._faces.reshape(-1)
            compressed.append( lz4.frame.compress(flat_faces) )
            self._faces = None

            # Compress twice: still fast, even smaller
            self._lz4_items = list(map(lz4.frame.compress, compressed))
        
        return sum(map(len, self._lz4_items))
    

    def _uncompress(self):
        if self._draco_bytes is not None:
            self._uncompress_from_draco()
        elif self._lz4_items is not None:
            self._uncompress_from_lz4()
        
        assert self._vertices_zyx is not None
        assert self._normals_zyx is not None
        assert self._faces is not None
    

    def _uncompress_from_draco(self):
        assert _dvidutils_available, \
            "Can't decode from draco if dvidutils isn't installed"
        vertices_xyz, normals_xyz, self._faces = decode_drc_bytes_to_faces(self._draco_bytes)
        self._vertices_zyx = vertices_xyz[:, ::-1]
        self._normals_zyx = normals_xyz[:, ::-1]
        self._draco_bytes = None
    

    def _uncompress_from_lz4(self):
        # Note: data was compressed twice, so uncompress twice
        uncompressed = list(map(lz4.frame.decompress, self._lz4_items))
        self._lz4_items = None

        decompress = lambda b: lz4.frame.decompress(b, return_bytearray=True)
        uncompressed = list(map(decompress, uncompressed))
        vertices_buf, normals_buf, faces_buf = uncompressed
        del uncompressed
        
        self._vertices_zyx = np.frombuffer(vertices_buf, np.float32).reshape((-1,3))
        del vertices_buf
        
        self._normals_zyx = np.frombuffer(normals_buf, np.float32).reshape((-1,3))
        del normals_buf

        self._faces = np.frombuffer(faces_buf, np.uint32).reshape((-1,3))
        del faces_buf

        # Should be writeable already
        self._vertices_zyx.flags['WRITEABLE'] = True
        self._normals_zyx.flags['WRITEABLE'] = True
        self._faces.flags['WRITEABLE'] = True


    def __getstate__(self):
        """
        Pickle representation.
        If pickle compression is enabled, compress the mesh to a buffer with draco,
        (or compress individual arrays with lz4) and discard the original arrays.
        """
        if self.pickle_compression_method:
            self.compress(self.pickle_compression_method)
        return self.__dict__

    def destroy(self):
        """
        Clear the mesh data.
        Release all of our big members.
        Useful for spark workflows, in which you don't immediately 
        all references to the mesh, but you know you're done with it.
        """
        self._draco_bytes = None
        self._vertices_zyx = None
        self._faces = None
        self._normals_zyx = None
        self._destroyed = True


    def auto_uncompress(f): # @NoSelf
        """
        Decorator.
        Before executing the decorated function, ensure that this mesh is not in a compressed state.
        """
        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            assert not self._destroyed
            if self._vertices_zyx is None:
                self._uncompress()
            return f(self, *args, **kwargs)
        return wrapper


    @property
    @auto_uncompress
    def vertices_zyx(self):
        return self._vertices_zyx

    @vertices_zyx.setter
    @auto_uncompress
    def vertices_zyx(self, new_vertices_zyx):
        self._vertices_zyx = new_vertices_zyx

    @property
    @auto_uncompress
    def faces(self):
        return self._faces

    @faces.setter
    @auto_uncompress
    def faces(self, new_faces):
        self._faces = new_faces

    @property
    @auto_uncompress
    def normals_zyx(self):
        return self._normals_zyx

    @normals_zyx.setter
    @auto_uncompress
    def normals_zyx(self, new_normals_zyx):
        self._normals_zyx = new_normals_zyx

    def sort_vertices(self):
        """
        Sort the vertex list lexicographically,
        while keeping the normals and faces arrays in sync.
        """
        order = np.lexsort(self.vertices_zyx.T[::-1])
        self.vertices_zyx = self.vertices_zyx[order]
        if len(self.normals_zyx) > 0:
            self.normals_zyx = self.normals_zyx[order]

        # The 'order' array can be thought of as a function from "sorted rank" to "original index":
        #   order[rank] -> original_index
        # If you argsort THAT, you get the inverse function[1], from "original index" to "sorted rank":
        #   inv_order[original_index] -> rank
        # We can use the latter to remap the IDs in self.faces
        #
        # [1]: Note that for any 1-D array A, the following holds:
        #   assert (argsort(A) == argsort(argsort(argsort(A)))).all()
        #
        inv_order = np.argsort(order).astype(np.uint32)
        self.faces = inv_order[self.faces]

    def stitch_adjacent_faces(self):
        """
        Identify duplicate vertices and remove them.
        Update vertex references in self.faces as needed to make sure no faces reference a deleted vertex.
        Also remove duplicate faces.
        Works in-place.

        Note: Normals are recomputed iff they were present originally.
        """
        # If we sort the vertices, finding duplicates is easy with np.diff
        self.sort_vertices()
        v = self.vertices_zyx
        non_dup = np.diff(v, axis=0, prepend=(v[:1] + 1)).any(axis=1)

        if non_dup.all():
            self.drop_duplicate_faces()
            return

        # Drop duplicate vertices
        self.vertices_zyx = self.vertices_zyx[non_dup]

        # Remap vertex IDs in the faces array to match the new vertices.
        # Since we sorted the vertices above, the remap array is simply an
        # array of consecutive integers, but with "forward-fill"
        # in the rows corresponding to duplicates.
        # We simulate a pandas DataFrame.groupby().ffill() using np.maximum.accumulate()
        remap = np.zeros(len(v), dtype=np.uint32)
        remap[non_dup] = np.arange(len(self.vertices_zyx))
        remap = np.maximum.accumulate(remap)
        self.faces = remap[self.faces]

        # Deduplicating vertices might reveal duplicated faces
        self.drop_duplicate_faces()

        if len(self.normals_zyx) > 0:
            self.recompute_normals(True)

    def drop_duplicate_faces(self):
        # Normalize face vertex order before checking for duplicates.
        # Technically, this means we don't distinguish
        # betweeen clockwise/counter-clockwise ordering,
        # but that seems unlikely to be a problem in practice.
        f = np.sort(self.faces, axis=1)
        order = np.lexsort(f.T[::-1])
        f = f[order]
        not_dup = np.diff(f, axis=0, prepend=(f[:1] + 1)).any(axis=1)
        self.faces = self.faces[order][not_dup]

    def recompute_normals(self, remove_degenerate_faces=True):
        """
        Compute the normals for this mesh.

        remove_degenerate_faces:
            If True, faces with no area (i.e. just lines) will be removed.
            (They have no effect on the vertex normals either way.)
        """
        face_normals = compute_face_normals(self.vertices_zyx, self.faces)

        if remove_degenerate_faces:
            # Degenerate faces ended up with a normal of 0,0,0.  Remove those faces.
            # (Technically, we might be left with unused vertices after this,
            #  but removing them requires relabeling the faces.
            #  Call stitch_adjacent_faces() if you want to remove them.)
            good_faces = face_normals.any(axis=1)
            if not good_faces.all():
                self.faces = self.faces[good_faces, :]
                face_normals = face_normals[good_faces, :]
            del good_faces

        if len(self.faces) == 0:
            # No faces left. Discard all remaining vertices and normals.
            self.vertices_zyx = np.zeros((0,3), np.float32)
            self.normals_zyx = np.zeros((0,3), np.float32)
        else:
            self.normals_zyx = compute_vertex_normals(self.vertices_zyx, self.faces, face_normals=face_normals)
        

    def simplify(self, fraction, in_memory=False, timeout=None):
        """
        Simplify this mesh in-place, by the given fraction (of the original vertex count).
        
        Note: timeout only applies to the NON-in-memory case.
        """
        # The fq-mesh-simplify tool rejects inputs that are too small (if the decimated face count would be less than 4).
        # We have to check for this in advance because we can't gracefully handle the error.
        # https://github.com/neurolabusc/Fast-Quadric-Mesh-Simplification-Pascal-/blob/master/c_code/Main.cpp
        if fraction is None or fraction == 1.0 or (len(self.faces) * fraction <= 4):
            if self.normals_zyx.shape[0] == 0:
                self.recompute_normals(True)
            return

        if in_memory:
            obj_bytes = write_obj(self.vertices_zyx[:,::-1], self.faces)
            bytes_stream = BytesIO(obj_bytes)
    
            simplify_input_pipe = TemporaryNamedPipe('input.obj')
            simplify_input_pipe.start_writing_stream(bytes_stream)
        
            simplify_output_pipe = TemporaryNamedPipe('output.obj')
        
            cmd = f'fq-mesh-simplify {simplify_input_pipe.path} {simplify_output_pipe.path} {fraction}'
            proc = subprocess.Popen(cmd, shell=True)
            mesh_stream = simplify_output_pipe.open_stream('rb')
            
            # The fq-mesh-simplify tool does not compute normals.
            vertices_xyz, self.faces, _empty_normals = read_obj(mesh_stream)
            self.vertices_zyx = vertices_xyz[:,::-1]
            mesh_stream.close()
    
            proc.wait(timeout=1.0)
            if proc.returncode != 0:
                msg = f"Child process returned an error code: {proc.returncode}.\n"\
                      f"Command was: {cmd}"
                logger.error(msg)
                raise RuntimeError(msg)
        else:
            obj_dir = AutoDeleteDir()
            undecimated_path = f'{obj_dir}/undecimated.obj'
            decimated_path = f'{obj_dir}/decimated.obj'
            write_obj(self.vertices_zyx[:,::-1], self.faces, output_file=undecimated_path)
            cmd = f'fq-mesh-simplify {undecimated_path} {decimated_path} {fraction}'
            subprocess.check_call(cmd, shell=True, timeout=timeout)
            with open(decimated_path, 'rb') as decimated_stream:
                # The fq-mesh-simplify tool does not compute normals.
                vertices_xyz, self.faces, _empty_normals = read_obj(decimated_stream)
                self.vertices_zyx = vertices_xyz[:,::-1]

        # Force normal reomputation to eliminate possible degenerate faces
        # (Can decimation produce degenerate faces?)
        self.recompute_normals(True)


    def simplify_openmesh(self, fraction):
        """
        Simplify this mesh in-place, by the given fraction (of the original vertex count).
        Uses OpenMesh to perform the decimation.
        This has similar performance to our default simplify() method,
        but does not require a subprocess or conversion to OBJ.
        Therefore, it can be faster in cases where I/O is the major bottleneck,
        rather than the decimation procedure itself.
        (For example, when lightly decimating a large mesh, I/O is the bottleneck.)
        """
        if len(self.vertices_zyx) == 0:
            return

        target = max(4, int(fraction * len(self.vertices_zyx)))
        if fraction is None or fraction == 1.0:
            if len(self.normals_zyx) == 0:
                self.recompute_normals(True)
            return

        import openmesh as om

        # Mesh construction in OpenMesh produces a lot of noise on stderr.
        # Send it to /dev/null
        try:
            sys.stderr.fileno()
        except:
            # Can't redirect stderr if it has no file descriptor.
            # Just let the output spill to wherever it's going.
            m = om.TriMesh(self.vertices_zyx[:, ::-1], self.faces)
        else:
            # Hide stderr, since OpenMesh construction is super noisy.
            with stdout_redirected(stdout=sys.stderr):
                m = om.TriMesh(self.vertices_zyx[:, ::-1], self.faces)

        h = om.TriMeshModQuadricHandle()
        d = om.TriMeshDecimater(m)
        d.add(h)
        d.module(h).unset_max_err()
        d.initialize()

        logger.debug(f"Attempting to decimate to {target} (Reduce by {len(self.vertices_zyx) - target})")
        eliminated_count = d.decimate_to(target)
        logger.debug(f"Reduced by {eliminated_count}")
        m.garbage_collection()

        self.vertices_zyx = m.points()[:, ::-1].astype(np.float32)
        self.faces = m.face_vertex_indices().astype(np.uint32)

        # Force normal reomputation to eliminate possible degenerate faces
        # (Can decimation produce degenerate faces?)
        self.recompute_normals(True)


    def laplacian_smooth(self, iterations=1):
        """
        Smooth the mesh in-place.

        This is simplest mesh smoothing technique, known as Laplacian Smoothing.
        Relocates each vertex by averaging its position with those of its adjacent neighbors.
        Repeat for N iterations.

        Disadvantage: Results in overall shrinkage of the mesh, especially for many iterations.
                      (But nearly all smoothing techniques cause at least some shrinkage.)

        Normals are automatically recomputed, and 'degenerate' faces after smoothing are discarded.

        Args:
            iterations:
                How many passes to take over the data.
                More iterations results in a smoother mesh, but more shrinkage (and more CPU time).

        TODO: Variations of this technique can give refined results.
            - Try weighting the influence of each neighbor by it's distance to the center vertex.
            - Try smaller displacement steps for each iteration
            - Try switching between 'push' and 'pull' iterations to avoid shrinkage
            - Try smoothing "boundary" meshes independently from the rest of the mesh (less shrinkage)
            - Try "Cotangent Laplacian Smoothing"
        """
        if iterations == 0:
            if self.normals_zyx.shape[0] == 0:
                self.recompute_normals(True)
            return

        # Always discard old normals
        self.normals_zyx = np.zeros((0,3), np.float32)

        # Compute the list of all unique vertex adjacencies
        edges = np.concatenate( [self.faces[:, (0,1)],
                                 self.faces[:, (1,2)],
                                 self.faces[:, (2,0)]] )

        # Drop duplicates
        edges.sort(axis=1)
        edges = edges[np.lexsort(edges.T)]
        non_dups = np.diff(edges, axis=0, prepend=(edges[:1] + 1)).any(axis=1)
        edges = edges[non_dups]

        # How many neighbors for each vertex == how many times it is mentioned in the edge list
        neighbor_counts = np.bincount(edges.ravel(), minlength=len(self.vertices_zyx))

        new_vertices_zyx = np.empty_like(self.vertices_zyx)
        for _ in range(iterations):
            new_vertices_zyx[:] = self.vertices_zyx

            # For the complete edge index list, accumulate (sum) the vertexes on
            # the right side of the list into the left side's address and vice-versa.
            #
            ## We want something like this:
            # v1_indexes, v2_indexes = df['v1_id'], df['v2_id']
            # new_vertices_zyx[v1_indexes] += self.vertices_zyx[v2_indexes]
            # new_vertices_zyx[v2_indexes] += self.vertices_zyx[v1_indexes]
            #
            # ...but that doesn't work because v1_indexes will contain repeats,
            #    and "fancy indexing" behavior is undefined in that case.
            #
            # Instead, it turns out that np.ufunc.at() works (it's an "unbuffered" operation)
            np.add.at(new_vertices_zyx, edges[:, 0], self.vertices_zyx[edges[:, 1], :])
            np.add.at(new_vertices_zyx, edges[:, 1], self.vertices_zyx[edges[:, 0], :])

            # Here, '+1' because each point itself is included in the sum
            new_vertices_zyx[:] /= (neighbor_counts[:, None] + 1)

            # Swap (save RAM allocation overhead by reusing the new_vertices_zyx array between iterations)
            self.vertices_zyx, new_vertices_zyx = new_vertices_zyx, self.vertices_zyx

        # Smoothing can cause degenerate faces,
        # particularly in some small special cases like this:
        #
        #   1        1
        #  / \       |
        # 2---3 ==>  X (where X is occupied by both 2 and 3)
        #  \ /       |
        #   4        4
        #
        # Detecting and removing such degenerate faces is easy if we recompute the normals.
        # (If we don't remove them, draco chokes on them.)
        self.recompute_normals(True)
        assert self.normals_zyx.shape == self.vertices_zyx.shape


    def serialize(self, path=None, fmt=None):
        """
        Serialize the mesh data in either .obj, .drc, or .ngmesh format.
        If path is given, write to that file.
        Otherwise, return the serialized data as a bytes object.
        """
        if path is not None:
            fmt = os.path.splitext(path)[1][1:]
        elif fmt is None:
            fmt = 'obj'
            
        assert fmt in self.MESH_FORMATS, f"Unknown format: {fmt}"

        # Shortcut for empty mesh
        # Returns an empty buffer regardless of output format        
        empty_mesh = (self._draco_bytes is not None and self._draco_bytes == b'') or len(self.vertices_zyx)
        if empty_mesh == 0:
            if path:
                open(path, 'wb').close()
                return
            return b''

        if fmt == 'obj':
            if path:
                with open(path, 'wb') as f:
                    write_obj(self.vertices_zyx[:,::-1], self.faces, self.normals_zyx[:,::-1], f)
            else:
                return write_obj(self.vertices_zyx[:,::-1], self.faces, self.normals_zyx[:,::-1])

        elif fmt == 'drc':
            assert _dvidutils_available, \
                "Can't use draco compression if dvidutils isn't installed"
            draco_bytes = self._draco_bytes
            if draco_bytes is None:
                if self.normals_zyx.shape[0] == 0:
                    self.recompute_normals(True) # See comment in Mesh.compress()
                draco_bytes = encode_faces_to_drc_bytes(self.vertices_zyx[:,::-1], self.normals_zyx[:,::-1], self.faces)
            
            if path:
                with open(path, 'wb') as f:
                    f.write(draco_bytes)
            else:
                return draco_bytes
        elif fmt == 'ngmesh':
            if path:
                write_ngmesh(self.vertices_zyx[:,::-1], self.faces, path)
            else:
                return write_ngmesh(self.vertices_zyx[:,::-1], self.faces)


    @classmethod
    def concatenate_meshes(cls, meshes, keep_normals=True):
        """
        Combine the given list of Mesh objects into a single Mesh object,
        renumbering the face vertices as needed, and expanding the bounding box
        to encompass the union of the meshes.
        
        Args:
            meshes:
                iterable of Mesh objects
            keep_normals:
                If False, discard all normals
                It True:
                    If no meshes had normals, the result has no normals.
                    If all meshes had normals, the result preserves them.
                    It is an error to provide a mix of meshes that do and do not contain normals.
        Returns:
            Mesh
        """
        return concatenate_meshes(meshes, keep_normals)


def concatenate_meshes(meshes, keep_normals=True):
    """
    Combine the given list of Mesh objects into a single Mesh object,
    renumbering the face vertices as needed, and expanding the bounding box
    to encompass the union of the meshes.
    
    Args:
        meshes:
            iterable of Mesh objects
        keep_normals:
            If False, discard all normals
            It True:
                If no meshes had normals, the result has no normals.
                If all meshes had normals, the result preserves them.
                It is an error to provide a mix of meshes that do and do not contain normals.
    Returns:
        Mesh
    """
    if not isinstance(meshes, list):
        meshes = list(meshes)

    vertex_counts = np.fromiter((len(mesh.vertices_zyx) for mesh in meshes), np.int64, len(meshes))
    face_counts = np.fromiter((len(mesh.faces) for mesh in meshes), np.int64, len(meshes))

    if keep_normals:
        _verify_concatenate_inputs(meshes, vertex_counts)
        concatenated_normals = np.concatenate( [mesh.normals_zyx for mesh in meshes] )
    else:
        concatenated_normals = None

    # vertices and normals are simply concatenated
    concatenated_vertices = np.concatenate( [mesh.vertices_zyx for mesh in meshes] )
    
    # Faces need to be renumbered so that they refer to the correct vertices in the combined list.
    concatenated_faces = np.ndarray((face_counts.sum(), 3), np.uint32)

    vertex_offsets = np.add.accumulate(vertex_counts[:-1])
    vertex_offsets = np.insert(vertex_offsets, 0, [0])

    face_offsets = np.add.accumulate(face_counts[:-1])
    face_offsets = np.insert(face_offsets, 0, [0])
    
    for faces, face_offset, vertex_offset in zip((mesh.faces for mesh in meshes), face_offsets, vertex_offsets):
        concatenated_faces[face_offset:face_offset+len(faces)] = faces + vertex_offset

    # bounding box is just the min/max of all bounding coordinates.
    all_boxes = np.stack([mesh.box for mesh in meshes])
    total_box = np.array( [ all_boxes[:,0,:].min(axis=0),
                            all_boxes[:,1,:].max(axis=0) ] )

    return Mesh( concatenated_vertices, concatenated_faces, concatenated_normals, total_box )


def _verify_concatenate_inputs(meshes, vertex_counts):
    normals_counts = np.fromiter((len(mesh.normals_zyx) for mesh in meshes), np.int64, len(meshes))
    if not normals_counts.any() or (vertex_counts == normals_counts).all():
        # Looks good
        return

    # Uh-oh, we have a problem:
    # Either some meshes have normals while others don't, or some meshes
    # have normals that don't even match their OWN vertex count!
        
    import socket
    hostname = socket.gethostname()

    mismatches = (vertex_counts != normals_counts).nonzero()[0]

    msg = ("Mesh normals do not correspond to vertices.\n"
           "(Either exclude all normals, more make sure they match the vertices in every mesh.)\n"
           f"There were {len(mismatches)} mismatches out of {len(meshes)}\n")

    bad_mismatches = (normals_counts != vertex_counts) & (normals_counts != 0)
    if bad_mismatches.any():
        # Mismatches where the normals and vertices didn't even line up in the same mesh.
        # This should never happen.
        first_bad_mismatch = bad_mismatches.nonzero()[0][0]
        mesh = meshes[first_bad_mismatch]
        output_path = f'/tmp/BAD-mismatched-mesh-v{mesh.vertices_zyx.shape[0]}-n{mesh.normals_zyx.shape[0]}-{first_bad_mismatch}.obj'
        mesh.serialize(output_path)
        msg += f"Wrote first BAD mismatched mesh to {output_path} (host: {hostname})\n"
    
    missing_normals = (normals_counts != vertex_counts) & (normals_counts == 0)
    if missing_normals.any():
        # Mismatches where the normals and vertices didn't even line up in the same mesh.
        # This should never happen.
        first_missing_normals = missing_normals.nonzero()[0][0]
        output_path = f'/tmp/mismatched-mesh-no-normals-{first_missing_normals}.obj'
        meshes[first_missing_normals].serialize(output_path)
        msg += f"Wrote first mismatched (missing normals) mesh to {output_path} (host: {hostname})\n"
    
    matching_meshes = (normals_counts == vertex_counts) & (normals_counts > 0)
    if matching_meshes.any():
        first_matching_mesh = matching_meshes.nonzero()[0][0]
        output_path = f'/tmp/first-matching-mesh-{first_matching_mesh}.obj'
        meshes[first_matching_mesh].serialize(output_path)
        msg += f"Wrote first matching mesh to {output_path} (host: {hostname})\n"
    
    raise RuntimeError(msg)
