# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import trimesh


def remesh_mesh(mesh_path, remesh_path):
    mesh_simplify_trimesh(mesh_path, remesh_path)


def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
    # Load mesh directly with trimesh (handles .obj, .glb, etc.)
    # The original code used pymeshlab only for a format round-trip (load→save)
    # before handing off to trimesh for simplification. pymeshlab segfaults on
    # macOS Apple Silicon with this mesh, so we use trimesh throughout.
    mesh = trimesh.load(inputpath, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        # If it's a scene, merge into one mesh
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        ) if hasattr(mesh, "geometry") else mesh

    face_num = mesh.faces.shape[0]
    if face_num > target_count:
        mesh = mesh.simplify_quadric_decimation(target_count)

    # Always export as OBJ (strip any .glb suffix confusion)
    out_path = outputpath.replace(".glb", ".obj")
    mesh.export(out_path)
