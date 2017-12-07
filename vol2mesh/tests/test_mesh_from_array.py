import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

from vol2mesh import mesh_from_array

class Test_mesh_from_array(unittest.TestCase):
     
    def test(self):
        # Create a test object (shaped like an 'X')
        center_line_img = np.zeros((100,100,100), dtype=np.uint32)
        for i in range(100):
            center_line_img[i, i, i] = 1
            center_line_img[99-i, i, i] = 1
         
        # Scipy distance_transform_edt conventions are opposite of vigra:
        # it calculates distances of non-zero pixels to the zero pixels.
        center_line_img = 1 - center_line_img
        distance_to_line = distance_transform_edt(center_line_img)
        binary_vol = (distance_to_line <= 10).astype(np.uint8)

        #binary_vol = np.pad(binary_vol, 1, 'constant', constant_values=0)
        box = [(0,0,0), (binary_vol.shape)]

        # Does it run at all?
        mesh = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=None, step_size=1 )

        #with open('/tmp/test-mesh.obj', 'wb') as f:
        #    f.write(mesh)
  
        # Simplifying makes the file smaller
        mesh_simplified = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=0.5, step_size=1 )
        assert len(mesh) > len(mesh_simplified), f"not true: {len(mesh)} > {len(mesh_simplified)}"
        
        # Simplifying more makes it even smaller
        mesh_more_simplified = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=0.2, step_size=1 )
        assert len(mesh_simplified) > len(mesh_more_simplified), f"not true: {len(mesh_simplified)} > {len(mesh_more_simplified)}"
  
        # Coarser step size (no simplification) makes the file... smaller?
        mesh_coarse = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=None, step_size=5 )
        assert len(mesh) > len(mesh_coarse), f"not true: {len(mesh)} > {len(mesh_coarse)}"

        #with open('/tmp/test-coarse-mesh.obj', 'wb') as f:
        #    f.write(mesh_coarse)

        # Running draco on top should be smaller than simplification
        mesh_compressed = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=None, step_size=1, output_format='drc' )
        assert len(mesh_compressed) < len(mesh_more_simplified), "Draco-encoded mesh should be smaller"
        
        # Running both simplification and draco: even smaller 
        mesh_simple_compressed = mesh_from_array( binary_vol, box[0], 1, simplify_ratio=0.2, step_size=1, output_format='drc' )
        assert len(mesh_simple_compressed) < len(mesh_compressed), "Draco-encoded mesh should be smaller"


    def test_tiny_array(self):
        """
        The scikit-image marching cubes function complains if we give it volumes smaller than 2x2x2.
        But we support volumes down to 1x1x1...
        """
        one_voxel = np.ones((1,1,1), np.uint8)
        tiny_mesh = mesh_from_array( one_voxel, (0,0,0), 1, simplify_ratio=None, step_size=1 )
        assert len(tiny_mesh.decode().splitlines()) > 3
        
if __name__ == "__main__":
    unittest.main()
