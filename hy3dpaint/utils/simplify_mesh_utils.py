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


def remesh_mesh(mesh_path, remesh_path, budget=None):
    # Phase 5: accept MeshBudget; target_count defaults to budget.target_faces
    # when provided, else the legacy 40000.
    target_count = 40000
    if budget is not None and hasattr(budget, "target_faces"):
        target_count = int(budget.target_faces)
    mesh_simplify_trimesh(mesh_path, remesh_path, target_count=target_count)


def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
    # First, remove disconnected faces
    ms = pymeshlab.MeshSet()
    if inputpath.endswith(".glb"):
        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(inputpath)
    ms.save_current_mesh(outputpath.replace(".glb", ".obj"), save_textures=False)
    # Run the face-reduction routine
    courent = trimesh.load(outputpath.replace(".glb", ".obj"), force="mesh")
    face_num = courent.faces.shape[0]

    if face_num > target_count:
        mesh = mesh.simplify_quadric_decimation(target_count)

    # Always export as OBJ (strip any .glb suffix confusion)
    out_path = outputpath.replace(".glb", ".obj")
    mesh.export(out_path)
