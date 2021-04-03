import pytest
import copy
from itertools import starmap
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt

from vol2mesh.mesh import Mesh, concatenate_meshes

import faulthandler
faulthandler.enable()

try:
    import skimage
    _skimage_available = True
except ImportError:
    _skimage_available = False
    

def box_to_slicing(start, stop):
    """
    For the given bounding box (start, stop),
    return the corresponding slicing tuple.

    Example:
    
        >>> assert bb_to_slicing([1,2,3], [4,5,6]) == np.s_[1:4, 2:5, 3:6]
    """
    return tuple( starmap( slice, zip(start, stop) ) )


@pytest.fixture(scope='module')
def binary_vol_input():
    # Create a test object (shaped like an 'X')
    center_line_img = np.zeros((98,98,98), dtype=np.uint32)
    for i in range(98):
        center_line_img[i, i, i] = 1
        center_line_img[97-i, i, i] = 1

    # Scipy distance_transform_edt conventions are opposite of vigra:
    # it calculates distances of non-zero pixels to the zero pixels.
    center_line_img = 1 - center_line_img
    distance_to_line = distance_transform_edt(center_line_img)
    binary_vol = (distance_to_line <= 10).astype(np.uint8)

    binary_vol = np.pad(binary_vol, 1, 'constant')
    assert binary_vol.shape == (100,100,100)

    #binary_vol = np.pad(binary_vol, 1, 'constant', constant_values=0)
    data_box = [(0,0,0), (binary_vol.shape)]
    
    binary_vol = binary_vol
    data_box = data_box
    
    min_nonzero_coord = np.transpose(binary_vol.nonzero()).min(axis=0)
    max_nonzero_coord = np.transpose(binary_vol.nonzero()).max(axis=0)
    
    nonzero_box = np.array( [min_nonzero_coord, 1+max_nonzero_coord] )

    return binary_vol, data_box, nonzero_box


def test_basic(binary_vol_input):
    binary_vol, data_box, nonzero_box = binary_vol_input
    # Pretend the data was downsampled and translated,
    # and therefore the mesh requires upscaling and translation
    data_box = np.array(data_box)
    data_box += 1000
    
    nonzero_box = nonzero_box + 1000
    
    FACTOR = 2
    data_box *= FACTOR
    nonzero_box *= FACTOR
    
    mesh = Mesh.from_binary_vol( binary_vol, data_box )
    assert mesh.vertices_zyx.dtype == np.float32
    
    mesh_box = np.array([mesh.vertices_zyx.min(axis=0),
                         mesh.vertices_zyx.max(axis=0)])
    assert (mesh_box == nonzero_box).all(), f"{mesh_box.tolist()} != {nonzero_box.tolist()}"
    
    serialized = mesh.serialize(fmt='obj')
    unserialized = mesh.from_buffer(serialized, 'obj')
    assert len(unserialized.vertices_zyx) == len(mesh.vertices_zyx)
    
    serialized = mesh.serialize(fmt='drc')
    unserialized = mesh.from_buffer(serialized, 'drc')
    assert len(unserialized.vertices_zyx) == len(mesh.vertices_zyx)

    mesh = Mesh.from_binary_vol( binary_vol, data_box, mesh_origin = np.asarray(data_box[0]), fragment_shape = np.asarray(data_box[1]), fragment_origin = np.asarray(data_box[0]))
    serialized = mesh.serialize(fmt='custom_drc')
    unserialized = mesh.from_buffer(serialized, 'drc')
    assert len(unserialized.vertices_zyx) == len(mesh.vertices_zyx)

    serialized = mesh.serialize(fmt='ngmesh')
    unserialized = mesh.from_buffer(serialized, 'ngmesh')
    assert len(unserialized.vertices_zyx) == len(mesh.vertices_zyx)
    

def test_blockwise(binary_vol_input):
    binary_vol, data_box, nonzero_box = binary_vol_input
    data_box = np.array(data_box)
    blocks = []
    boxes = []
    for z in range(0,100,20):
        for y in range(0,100,20):
            for x in range(0,100,20):
                OVERLAP = 1
                box = np.asarray([(z,y,x), (z+20, y+20, x+20)], dtype=int)
                box[0] -= OVERLAP
                box[1] += OVERLAP
                box = np.maximum(box, 0)
                box = np.minimum(box, 1+data_box[1])

                block = binary_vol[box_to_slicing(*box)]
                if block.any():
                    blocks.append(block)
                    boxes.append( box )
    
    mesh = Mesh.from_binary_blocks(blocks, boxes)
    data_box = np.array(data_box)
    mesh_box = np.array([mesh.vertices_zyx.min(axis=0), mesh.vertices_zyx.max(axis=0)])
    assert (mesh_box == nonzero_box).all(), f"{mesh_box.tolist()} != {nonzero_box.tolist()}"

#         with open('/tmp/test-mesh.obj', 'wb') as f:
#             f.write(mesh.serialize())
# 
#         mesh.simplify(0.01)
#         with open('/tmp/test-mesh-simplified.obj', 'wb') as f:
#             f.write(mesh.serialize())
#         
#         with open('/tmp/test-mesh-simplified.drc', 'wb') as f:
#             f.write(mesh.serialize(fmt='drc'))

def test_blockwise_simple():
    """
    Simple test case to manually explore the output
    of marching cubes as computed in blocks without halo.
    """
    _ = 0
    img = [[_,_,_,_, _,1,_,_],
           [_,1,_,_, _,_,_,_],
           [_,_,1,1, 1,1,1,_],
           [_,1,1,1, 1,1,1,_],

           [_,1,1,1, 1,1,1,_],
           [_,1,1,1, 1,1,1,_],
           [_,1,1,1, 1,1,1,_],
           [_,_,_,_, _,_,_,_]]
    
    vol = np.zeros((3,8,8), dtype=bool)
    vol[1] = img

    blocks = (vol[:, 0:4, 0:4],
              vol[:, 0:4, 4:8],
              vol[:, 4:8, 0:4],
              vol[:, 4:8, 4:8])
    
    starts = [[0,0,0],
              [0,0,4],
              [0,4,0],
              [0,4,4]]
    
    starts = np.array(starts)
    boxes = np.zeros((4,2,3), np.uint32)
    boxes[:,0,:] = starts
    boxes[:,1,:] = starts + (3,4,4)

    _mesh = Mesh.from_binary_blocks(blocks[3:4], boxes[3:4], stitch=False)
    #_mesh.serialize('/tmp/simple-blocks.obj')
    
    #print(np.asarray(sorted(_mesh.vertices_zyx.tolist())))
    
@pytest.mark.skipif(not _skimage_available, reason="Skipping skimage-based tests")
def test_tiny_array():
    """
    Tiny arrays trigger an exception in skimage, so they must be padded first.
    Verify that they can be meshified (after padding).
    """
    one_voxel = np.ones((1,1,1), np.uint8)
    _mesh = Mesh.from_binary_vol( one_voxel, [(0,0,0), (1,1,1)], method='skimage' )
    _mesh = Mesh.from_binary_vol( one_voxel, [(0,0,0), (1,1,1)], method='ilastik' )


def test_solid_array():
    """
    Solid volumes can't be meshified. An empty mesh is returned instead.
    """
    box = [(0,0,0), (3,3,3)]
    solid_volume = np.ones((3,3,3), np.uint8)
    
    mesh = Mesh.from_binary_vol( solid_volume, box )
    assert mesh.vertices_zyx.shape == (0,3)
    assert mesh.faces.shape == (0,3)
    assert mesh.normals_zyx.shape == (0,3)
    assert (mesh.box == box).all()


def test_empty_mesh():
    """
    What happens when we call functions on an empty mesh?
    """
    mesh = Mesh( np.zeros((0,3), np.float32), np.zeros((0,3), int) )

    mesh.simplify(1.0)
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.simplify(0.1)
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.simplify_openmesh(1.0)
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.simplify_openmesh(0.1)
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.laplacian_smooth(0)
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.laplacian_smooth(2)
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.stitch_adjacent_faces()
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.serialize(fmt='obj')
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.serialize(fmt='drc')
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.serialize(fmt='custom_drc')
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0

    mesh.compress()
    concatenate_meshes((mesh, mesh))
    assert len(mesh.vertices_zyx) == len(mesh.normals_zyx) == len(mesh.faces) == 0


def test_smoothing_trivial():
    vertices_zyx = np.array([[0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.0, 0.0, 2.0]])

    # This "face" is actually straight line,
    # which makes it easy to see what's going on
    faces = np.array([[0,1,2]])
    mesh = Mesh(vertices_zyx, faces)
    average_vertex = vertices_zyx.sum(axis=0) / 3
    mesh.laplacian_smooth(1)
    assert (mesh.vertices_zyx == average_vertex).all()


def test_smoothing_hexagon():
    """
    Try 'smoothing' a simple 2D hexagon, which is an easy case to understand.
    """
    # This map is correctly labeled with the vertex indices
    _ = -1
    hexagon = [[[_,_,_,_,_,_,_],
                [_,_,0,_,1,_,_],
                [_,_,_,_,_,_,_],
                [_,2,_,3,_,4,_],
                [_,_,_,_,_,_,_],
                [_,_,5,_,6,_,_],
                [_,_,_,_,_,_,_]]]

    hexagon = 1 + np.array(hexagon)
    original_vertices_zyx = np.transpose(hexagon.nonzero())
    faces = [[3,1,4],
             [3,4,6],
             [3,6,5],
             [3,5,2],
             [3,2,0],
             [3,0,1]]
    
    mesh = Mesh(original_vertices_zyx, faces)
    #mesh.serialize('/tmp/hexagon.obj')

    mesh.laplacian_smooth(1)
    #mesh.serialize('/tmp/hexagon-smoothed.obj')
    
    # Since vertex 3 is exactly centered between the rest,
    # it's location never changes.
    assert  (mesh.vertices_zyx[3] == original_vertices_zyx[3]).all()


def test_smoothing_X(binary_vol_input):
    """
    This just exercises the code on our standard X-shaped
    test object, but doesn't verify the results.
    
    Uncomment the serialize() lines to inspect the effects manually.
    """
    binary_vol, data_box, _nonzero_box = binary_vol_input
    mesh = Mesh.from_binary_vol( binary_vol, data_box )
    #mesh.serialize('/tmp/x-unsmoothed.obj')

    mesh.simplify(0.2)
    mesh.laplacian_smooth(5)        
    #mesh.serialize('/tmp/x-simplified-smoothed.obj')

    mesh = Mesh.from_binary_vol( binary_vol, data_box )
    mesh.laplacian_smooth(5)
    mesh.simplify(0.2)
    #mesh.serialize('/tmp/x-smoothed-simplified.obj')

    mesh = Mesh.from_binary_vol( binary_vol, data_box )
    mesh.laplacian_smooth(5)
    mesh.simplify_openmesh(0.2)
    #mesh.serialize('/tmp/x-smoothed-simplified.obj')


def test_stitch():
    vertices = np.zeros( (10,3), np.float32 )
    vertices[:,0] = np.arange(10)
    
    # Make 3 and 6 duplicates of 2 and 4, respectively
    vertices[3] = vertices[2]
    vertices[6] = vertices[4]
    
    faces = [[0,1,2],
             [3,4,5],
             [6,7,8],
             [7,8,4]] # <- duplicate face (different vertex order, and 4=6)
    
    # After mapping away from dupe vertices
    remapped_faces = [[0,1,2],
                      [2,4,5], 
                      [4,7,8],
                      [7,8,4]] # duplicated face (different vertex order)
    
    remapped_faces = np.array(remapped_faces)
    
    # After dropping dupe rows
    remapped_faces[(remapped_faces > 6)] -= 1
    remapped_faces[(remapped_faces > 3)] -= 1

    # Drop last face (duplicated)
    remapped_faces = remapped_faces[:-1, :]
    
    reduced_vertices = list(vertices)
    del reduced_vertices[9] # wasn't referenced to begin with
    del reduced_vertices[6] # was duplicated
    del reduced_vertices[3] # was duplicated
    reduced_vertices = np.asarray(reduced_vertices)
    
    mesh = Mesh(vertices, faces)
    mesh.stitch_adjacent_faces()
    
    assert (mesh.faces == remapped_faces).all()
    assert (mesh.vertices_zyx == reduced_vertices).all()


def test_pickling(binary_vol_input):
    binary_vol, _data_box, _nonzero_box = binary_vol_input
    mesh = Mesh.from_binary_vol( binary_vol )
    pickled = pickle.dumps(mesh)
    unpickled = pickle.loads(pickled)
    
    # It's not easy to verify that unpickled is identical,
    # since draco may re-order vertices and faces.
    # The validity of our draco encoding functions is tested elsewhere,
    # so here we just check for vertex/face count
    assert len(mesh.vertices_zyx) == len(unpickled.vertices_zyx)
    assert len(mesh.faces) == len(unpickled.faces)

def test_pickling_empty():
    mesh = Mesh(np.zeros((0,3), np.float32), 
                np.zeros((0,3), np.uint32))
    pickled = pickle.dumps(mesh)
    unpickled = pickle.loads(pickled)
    
    assert len(unpickled.vertices_zyx) == 0
    assert len(unpickled.faces) == 0


def test_normals_implementations(binary_vol_input):
    """
    Compare the numpy-based and numba-based normals computation implementations.
    """
    binary_vol, data_box, _nonzero_box = binary_vol_input

    try:
        from vol2mesh.normals import (compute_face_normals, compute_face_normals_numba, compute_face_normals_numpy, 
                                      compute_vertex_normals, compute_vertex_normals_numba, compute_vertex_normals_numpy)
    except ImportError:
        pytest.skip("numba not installed")

    mesh = Mesh.from_binary_vol( binary_vol, data_box )

    face_normals_default = compute_face_normals(mesh.vertices_zyx, mesh.faces, normalize=True)
    face_normals_numba = compute_face_normals_numba(mesh.vertices_zyx, mesh.faces, normalize=True)
    face_normals_numpy = compute_face_normals_numpy(mesh.vertices_zyx, mesh.faces, normalize=True)
    assert np.allclose(face_normals_numba, face_normals_default)
    assert np.allclose(face_normals_numba, face_normals_numpy)

    vertex_normals_default = compute_vertex_normals(mesh.vertices_zyx, mesh.faces, weight_by_face_area=False, face_normals=face_normals_default)
    vertex_normals_numba = compute_vertex_normals_numba(mesh.vertices_zyx, mesh.faces, weight_by_face_area=False, face_normals=face_normals_numba)
    vertex_normals_numpy = compute_vertex_normals_numpy(mesh.vertices_zyx, mesh.faces, weight_by_face_area=False, face_normals=face_normals_numpy)
    assert np.allclose(vertex_normals_numba, vertex_normals_default)
    assert np.allclose(vertex_normals_numba, vertex_normals_numpy)


def test_normals_guarantees(binary_vol_input):
    """
    Member functions have guarantees about whether normals are present or absent after the function runs.
    - simplify(): Always present afterwards
    - laplacian_smooth(): Always present afterwards
    - stitch_adjacent_faces(): Present afterwards IFF they were present before.
    """
    binary_vol, data_box, _nonzero_box = binary_vol_input
    data_box = np.array(data_box)
    
    FACTOR = 2
    data_box *= FACTOR
    
    mesh_orig = Mesh.from_binary_vol( binary_vol, data_box )
    
    mesh = copy.deepcopy(mesh_orig)
    assert mesh.normals_zyx.shape[0] > 1
    
    # Verify normals are always present after simplification,
    # Regardless of whether or not they were present before,
    # or if simplification was even performed.
    mesh.simplify(1.0)
    assert mesh.normals_zyx.shape[0] > 1

    mesh.simplify(0.5)
    assert mesh.normals_zyx.shape[0] > 1

    mesh.drop_normals()
    mesh.simplify(0.5)
    assert mesh.normals_zyx.shape[0] > 1

    # Verify normals are always present after smoothing,
    # Regardless of whether or not they were present before,
    # or if smoothing was even performed.
    mesh = copy.deepcopy(mesh_orig)
    mesh.laplacian_smooth(0)
    assert mesh.normals_zyx.shape[0] > 1

    mesh.laplacian_smooth(2)
    assert mesh.normals_zyx.shape[0] > 1

    mesh.drop_normals()
    mesh.laplacian_smooth(2)
    assert mesh.normals_zyx.shape[0] > 1

    # Verify that the presence or absence of normals is the SAME after stitching,
    # Whether or not stitching had any effect.
    
    # no stitching, keep normals
    mesh = copy.deepcopy(mesh_orig)
    stitching_performed = mesh.stitch_adjacent_faces()
    assert not stitching_performed
    assert mesh.normals_zyx.shape[0] > 1
    
    # no stitching, no normals in the first place
    mesh.drop_normals()
    stitching_performed = mesh.stitch_adjacent_faces()
    assert not stitching_performed
    assert mesh.normals_zyx.shape[0] == 0

    # stitching, generate normals
    mesh = copy.deepcopy(mesh_orig)
    duplicated_mesh = concatenate_meshes([mesh, mesh])
    assert duplicated_mesh.normals_zyx.shape[0] > 1
    stitching_performed = duplicated_mesh.stitch_adjacent_faces()
    assert stitching_performed
    assert duplicated_mesh.normals_zyx.shape[0] > 1
    
    # stitching, no normals in the first place
    mesh = copy.deepcopy(mesh_orig)
    duplicated_mesh = concatenate_meshes([mesh, mesh])
    duplicated_mesh.drop_normals()
    stitching_performed = duplicated_mesh.stitch_adjacent_faces()
    assert stitching_performed
    assert duplicated_mesh.normals_zyx.shape[0] == 0

def test_compress(binary_vol_input):
    binary_vol, data_box, _nonzero_box = binary_vol_input
    mesh_orig = Mesh.from_binary_vol( binary_vol, data_box )
    uncompressed_size = mesh_orig.normals_zyx.nbytes + mesh_orig.vertices_zyx.nbytes + mesh_orig.faces.nbytes
    mesh = copy.deepcopy(mesh_orig)
    
    size = mesh.compress('lz4')
    assert size < uncompressed_size
    assert (mesh.faces == mesh_orig.faces).all()
    assert (mesh.vertices_zyx == mesh_orig.vertices_zyx).all()
    assert (mesh.normals_zyx == mesh_orig.normals_zyx).all()

    # Draco is lossy, so we can't compare exactly.
    # Just make sure the arrays are at least of the correct shape.
    size = mesh.compress('draco')
    assert size < uncompressed_size
    assert (mesh.faces.shape == mesh_orig.faces.shape)
    assert (mesh.vertices_zyx.shape == mesh_orig.vertices_zyx.shape)
    assert (mesh.normals_zyx.shape == mesh_orig.normals_zyx.shape)


    mesh = copy.deepcopy(mesh_orig)
    mesh.mesh_origin = np.asarray(data_box[0])
    mesh.fragment_shape = np.asarray(data_box[1])
    mesh.fragment_origin = np.asarray(data_box[0])
    size = mesh.compress('custom_draco')
    assert size < uncompressed_size
    assert (mesh.faces.shape == mesh_orig.faces.shape)
    assert (mesh.vertices_zyx.shape == mesh_orig.vertices_zyx.shape)
    assert (mesh.normals_zyx.shape == mesh_orig.normals_zyx.shape)
    
@pytest.fixture(scope='module')
def tiny_meshes():
    vertexes_1 = np.array([[0,0,0],
                           [0,1,0],
                           [0,1,1]])

    faces_1 = np.array([[2,1,0]])

    vertexes_2 = np.array([[0,0,1],
                           [0,2,0],
                           [0,2,2]])

    faces_2 = np.array([[2,1,0],
                        [1,2,0]])

    vertexes_3 = np.array([[1,0,1],
                           [1,2,0],
                           [1,2,2]])

    faces_3 = np.array([[1,2,0]])

    mesh_1 = Mesh(vertexes_1, faces_1)
    mesh_2 = Mesh(vertexes_2, faces_2)
    mesh_3 = Mesh(vertexes_3, faces_3)
    mesh_4 = Mesh(np.zeros((0,3), np.float32), np.zeros((0,3), np.uint32)) # Empty mesh
    
    return mesh_1, mesh_2, mesh_3, mesh_4

def test_concatenate(tiny_meshes):
    mesh_1, mesh_2, mesh_3, mesh_4 = tiny_meshes
    vertexes_1, vertexes_2, vertexes_3 = mesh_1.vertices_zyx, mesh_2.vertices_zyx, mesh_3.vertices_zyx
    faces_1, faces_2, faces_3 = mesh_1.faces, mesh_2.faces, mesh_3.faces
    
    combined_mesh = concatenate_meshes((mesh_1, mesh_2, mesh_3, mesh_4))
    assert (combined_mesh.vertices_zyx == np.concatenate((vertexes_1, vertexes_2, vertexes_3))).all()
    
    expected_faces = np.concatenate((faces_1, faces_2, faces_3))
    expected_faces[len(faces_1):] += len(vertexes_1)
    expected_faces[len(faces_1)+len(faces_2):] += len(vertexes_2)

    assert (combined_mesh.faces == expected_faces).all()

def test_concatenate_with_normals(tiny_meshes):
    mesh_1, mesh_2, mesh_3, mesh_4 = tiny_meshes
    vertexes_1, vertexes_2, vertexes_3 = mesh_1.vertices_zyx, mesh_2.vertices_zyx, mesh_3.vertices_zyx
    faces_1, faces_2, faces_3 = mesh_1.faces, mesh_2.faces, mesh_3.faces

    mesh_1.recompute_normals()
    mesh_2.recompute_normals()
    mesh_3.recompute_normals()
    
    combined_mesh = concatenate_meshes((mesh_1, mesh_2, mesh_3, mesh_4))
    assert (combined_mesh.vertices_zyx == np.concatenate((vertexes_1, vertexes_2, vertexes_3))).all()
    
    expected_faces = np.concatenate((faces_1, faces_2, faces_3))
    expected_faces[len(faces_1):] += len(vertexes_1)
    expected_faces[len(faces_1)+len(faces_2):] += len(vertexes_2)

    assert (combined_mesh.faces == expected_faces).all()

def test_mismatches(tiny_meshes):
    mesh_1, mesh_2, mesh_3, _mesh_4 = tiny_meshes
    mesh_1.recompute_normals()
    
    # let's really mess up mesh 3 -- give it the wrong number of normals
    mesh_3.recompute_normals()
    mesh_3.normals_zyx = mesh_3.normals_zyx[:-1]

    try:
        _combined_mesh = concatenate_meshes((mesh_1, mesh_2, mesh_3))
    except RuntimeError:
        pass
    else:
        assert False, "Expected a RuntimeError, but did not see one."


if __name__ == "__main__":
    args = ['-s', '--tb=native', '--pyargs', __file__]
    #args += ['-k' 'normals_implementations']
    pytest.main(args)
