# vol2mesh

A mesh generating wrapper in Python 3, with optional simplification and draco compression.

Installation
------------

```bash
conda install -c flyem-forge vol2mesh
```

Python usage
------------

Example:

```python
from vol2mesh import Mesh, mesh_from_array, concatenate_meshes

# Creation from binary vol (via marching cubes)
# For some binary ndarray 'binary_array'
binary_vol = ...

box = [(0,0,0), (binary_vol.shape)]
mesh = Mesh.from_binary_vol( binary_vol, box )

# Alternative: Creation from file(s)/buffer
mesh2 = Mesh.from_file( '/path/to/mesh.drc' )
mesh3 = Mesh.from_directory( '/path/to/meshes/' )
mesh4 = Mesh.from_bytes( obj_bytes, 'obj' )

# Basic ops
mesh.laplacian_smooth(3)
mesh.simplify(0.2)

# alternative implementation of simplify(), based on OpenMesh
mesh.simplify_openmesh(0.2)

# Serialize to disk
mesh.serialize('/tmp/my-mesh.obj')
mesh.serialize('/tmp/my-mesh.drc')

# Serialize to buffer
mesh_bytes = mesh.serialize(fmt='drc')

# Combine meshes (with proper vertex renumbering in the faces)
combined_mesh = concatenate_meshes([mesh1, mesh2, mesh3])

# Optional: Deduplicate vertices
combined_mesh.stitch_adjacent_faces()

# Less common ops
mesh.drop_normals()
mesh.drop_unused_vertices()
mesh.recompute_normals()
```


Appendix: Dependencies
----------------------

- Mesh decimation is performed via the `pyfqmr-Fast-Quadric-Mesh-Reduction/tree/master` package:
   - https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction/tree/master
- We support the [draco] compressed mesh serialization format via [`DracoPy`][dracopy].
- The default marching cubes implementation is from the ilastik project's [`marching_cubes` library][marching_cubes].
  - Optionally, we support `skimage.marching_cubes_lewiner()` as an alternative, but you must install `scikit-image` yourself (it is not pulled in as a required dependency.


[dracopy]: https://github.com/seung-lab/DracoPy
[draco]: https://github.com/google/draco
[marching_cubes]: https://github.com/ilastik/marching_cubes
[numba]: https://numba.pydata.org
