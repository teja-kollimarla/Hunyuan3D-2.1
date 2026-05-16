# Data processing

This is the data-processing pipeline for 3D shape and texture generation.

**Notes**:
1. This implementation is a simplified version of our industrial pipeline.
2. The rendering script is based on [TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/blender_script/render.py).

## Rendering

### Motivation
The rendering script `render/render.py` has three main purposes:
1. Use Blender to convert complex 3D formats into PLY files for further processing.
2. Render conditional images for DiT training.
3. Render orthographic images, PBR materials, and conditional signals (world-space normals and positions) for texture generation.

### Requirements
The rendering script runs under Blender 4.1. You need to install `opencv`, `OpenEXR`, and `Imath` using Blender's bundled Python. Example on macOS:
```bash
/Applications/Blender.app/Contents/Resources/4.1/python/bin/python3.11 -m pip install OpenEXR Imath opencv-python
```

### Execution
The first two purposes can be accomplished with a single command:
```bash
$BLENDER_PATH -b -P render/render.py -- \
    --object ${INPUT_FILE} --geo_mode --resolution 512 \
    --output_folder $OUTPUT_FOLDER
```
For the third purpose, simply drop the `--geo_mode` flag.

## Watertight mesh processing and sampling

### Motivation
To learn the SDF representation used by 3DShape2VecSets, we need a watertight input mesh. This pipeline takes a raw triangle mesh and produces three required data types:
1. **Surface samples** — input points for the encoder.
2. **Volume samples** — query points for SDF evaluation in the decoder.
3. **Volume SDFs** — ground-truth signed-distance values for VAE training.

### Execution
Given a triangle mesh (OBJ/OFF), this produces:
1. A watertight mesh (`${OUTPUT_NAME}_watertight.obj`).
2. Surface-point samples (`${OUTPUT_NAME}_surface.npz`).
3. Volume samples with SDFs (`${OUTPUT_NAME}_sdf.npz`).

**Command:**
```bash
python3 watertight/watertight_and_sample.py \
    --input_obj ${INPUT_MESH} \
    --output_prefix ${OUTPUT_NAME}
```

### Output data format

#### 1. Surface samples (`${OUTPUT_NAME}_surface.npz`)
Contains two point-cloud arrays stored in numpy NPZ format:

| Key              | Shape    | Dtype     | Description                                |
|------------------|----------|-----------|--------------------------------------------|
| `random_surface` | `(N, 6)` | `float16` | Uniform point samples on the surface       |
| `sharp_surface`  | `(M, 6)` | `float16` | Samples near sharp edges of the mesh       |

#### 2. Volume SDF samples (`${OUTPUT_NAME}_sdf.npz`)
Contains three sample types stored as point/label array pairs. For each type `${type}`:

| Sample type    | Points array          | SDF label array        | Shape           | Dtype     | Description                       |
|----------------|-----------------------|------------------------|-----------------|-----------|-----------------------------------|
| `vol`          | `vol_points`          | `vol_label`            | `(P, 3)/(P,)`   | `float16` | Uniform samples in volume         |
| `random_near`  | `random_near_points`  | `random_near_label`    | `(Q, 3)/(Q,)`   | `float16` | Samples close to the surface      |
| `sharp_near`   | `sharp_near_points`   | `sharp_near_label`     | `(R, 3)/(R,)`   | `float16` | Samples close to sharp edges      |

**Data specifications**:
- All point coordinates (`*_points` arrays) contain 3D positions stored as `float16`.
- All SDF values (`*_label` arrays) are `float16` scalars where:
  - **Positive** — outside the surface.
  - **Negative** — inside the surface.
  - **Zero** — on the surface.
- Array dimensions:
  - `N`, `M`, `P`, `Q`, `R` are sample counts (vary per shape).
  - `3` is for XYZ coordinates.
  - `6` is for XYZ + normal coordinates.
- All arrays are stored uncompressed in numpy's NPZ format.

## Overall pipeline script
Edit these four variables in `pipeline.sh`:
1. **INPUT_FILE** — path to each 3D source asset.
2. **OUTPUT_FOLDER** — root path for the output dataset.
3. **NAME** — output naming for each data point.
4. **BLENDER_PATH** — path to the Blender executable.

Then run:
```bash
bash pipeline.sh
```
