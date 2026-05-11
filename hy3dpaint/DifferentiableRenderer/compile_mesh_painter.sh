#!/usr/bin/env bash
# Compile mesh_inpaint_processor pybind11 extension.
# On macOS, Python symbols are resolved at load time by the interpreter,
# so we need -undefined dynamic_lookup to avoid linker errors.
# Uses the active Python (from venv) for both includes and extension suffix.
set -e

PYTHON="${PYTHON:-python3}"

UNAME="$(uname)"
LDFLAGS=""
if [ "$UNAME" = "Darwin" ]; then
    LDFLAGS="-undefined dynamic_lookup"
fi

EXT_SUFFIX=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

c++ -O3 -Wall -shared -std=c++11 -fPIC $LDFLAGS \
    $("$PYTHON" -m pybind11 --includes) \
    mesh_inpaint_processor.cpp \
    -o "mesh_inpaint_processor${EXT_SUFFIX}"

echo "Built: mesh_inpaint_processor${EXT_SUFFIX}"
