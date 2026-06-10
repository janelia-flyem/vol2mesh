import re
from io import BytesIO
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd

def write_obj(vertices_xyz, faces, normals_xyz=None, output_file=None):
    """
    Generate an OBJ file from the given (binary) data and write it to the given byte stream or file path.
    
    If no output stream/path is given, return a bytes object containing the OBJ data.

    vertices_xyz: np.ndarray, shape=(N,3), dtype=float
    faces: np.ndarray, shape=(N,3), dtype=int
    normals_xyz: np.ndarray, shape=(N,3), dtype=float (Optional.)
    
    Note: Each 'face' consists of 3 indexes, which correspond to indexes in the vertices_xyz.
          The indexes should be 0-based. (They will be converted to 1-based in the OBJ)
    """
    if normals_xyz is None:
        normals_xyz = np.zeros((0,3), np.float32)

    need_close = True

    if output_file is None:
        mesh_bytestream = BytesIO()
    elif isinstance(output_file, (str, Path)):
        mesh_bytestream = open(output_file, 'wb')
    else:
        assert hasattr(output_file, 'write')
        mesh_bytestream = output_file
        need_close = False
    
    try:
        _write_obj(vertices_xyz, faces, normals_xyz, mesh_bytestream)
        if output_file is None:
            return mesh_bytestream.getvalue()
    finally:
        if need_close:
            mesh_bytestream.close()

def _write_obj(vertices_xyz, faces, normals_xyz, mesh_bytestream):
    """
    Given lists of vertices and faces, write them to the given stream in .obj format.
    
    vertices_xyz: np.ndarray, shape=(N,3), dtype=float
    faces: np.ndarray, shape=(N,3), dtype=int
    normals_xyz: np.ndarray, shape=(N,3), dtype=float
    
    Note: Each 'face' consists of 3 indexes, which correspond to indexes in the vertices_xyz.
          The indexes should be 0-based. (They will be converted to 1-based in the OBJ)
        
    Returns:
        BytesIO
    """
    if len(vertices_xyz) == 0:
        # Empty meshes result in no bytes
        return

    mesh_bytestream.write(b"# OBJ file\n")

    # Tips for faster exports
    # https://github.com/mikedh/trimesh/blob/main/trimesh/exchange/README.md

    mesh_bytestream.write(("v {:.7g} {:.7g} {:.7g}\n" * len(vertices_xyz)).format(*vertices_xyz.ravel()).encode('utf-8'))
    if len(normals_xyz) > 0:
        mesh_bytestream.write(("vn {:.7g} {:.7g} {:.7g}\n" * len(normals_xyz)).format(*normals_xyz.ravel()).encode('utf-8'))

    # OBJ format: Faces start at index 1 (not 0)
    faces = faces + 1
    if len(normals_xyz) > 0:
        faces = as_strided(faces, faces.shape + (2,), faces.strides + (0,), writeable=False)
        mesh_bytestream.write(("f {}//{} {}//{} {}//{}\n" * len(faces)).format(*faces.flat).encode('utf-8'))
    else:
        mesh_bytestream.write(("f {} {} {}\n" * len(faces)).format(*faces.ravel()).encode('utf-8'))


def read_obj(mesh_bytestream):
    """
    Read the OBJ file from the given file stream and return the vertexes/faces/normals as numpy arrays.
    
    Note:
        Many OBJ features, such as texture coordinates (vt lines), parameter space vertices (vp lines),
        "line elements", "materials", or faces with out-of-order vertex normal indices are not supported.
        
        f 20 30 40                 # <-- OK
        f 20//20 30//30 40//40     # <-- OK
        f 20/1/20 30/2/30 40/3/40  # <-- OK, but texture coordinates will be discarded
        f 20//1 30//2 40//3        # <-- NOT SUPPORTED (out-of-order vertex normals)
    
    Returns:
        vertices_xyz, faces, normals_xyz
        
        Note that the 'faces' indexes are 0-based
        (python conventions, not OBJ conventions, which start with 1)
    """
    if isinstance(mesh_bytestream, bytes):
        mesh_bytes = mesh_bytestream
    elif isinstance(mesh_bytestream, (str, Path)):
        mesh_bytes = open(mesh_bytestream, 'rb').read()
    else:
        mesh_bytes = mesh_bytestream.read()

    def strip_comments(code: str) -> str:
        return b'\n'.join(
            line for line in code.splitlines()
            if line.strip() and not line.lstrip().startswith(b'#')
        )
    mesh_bytes = strip_comments(mesh_bytes)

    # For faces, remove everything but the vertex index
    mesh_bytes = re.sub(rb'(\d+)/\S+', rb'\1', mesh_bytes)
    # Read as CSV
    df = pd.read_csv(BytesIO(mesh_bytes), sep=' ', header=None, names=['element', *'xyz'])
    vertices_xyz = df.query('element == "v"')[[*'xyz']].values.astype(np.float32)
    normals_xyz = df.query('element == "vn"')[[*'xyz']].values.astype(np.float32)
    faces = df.query('element == "f"')[[*'xyz']].values.astype(np.int32)

    # In OBJ, indices start at index 1 (not 0), but we want to use numpy conventions
    faces -= 1

    if len(faces) > 0 and faces.max() >= len(vertices_xyz):
        raise RuntimeError(f"Unexpected format: A face referenced vertex {faces.max()}, which is out-of-bounds for the vertex list.")

    return vertices_xyz, faces, normals_xyz
