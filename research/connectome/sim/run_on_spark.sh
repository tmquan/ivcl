#!/bin/bash
# Run on the DGX Spark. Fetches the data, makes the meshes, and renders both
# scenes in Isaac Sim without a window. Safe to run again: finished steps are
# skipped.
#
#   bash run_on_spark.sh            # everything
#   bash run_on_spark.sh crop313    # one scene: crop313 or cremi_A
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/IsaacSim/_build/linux-aarch64/release}"
DATA="${CONNECTOME_DATA:-$HOME/connectome/data}"
OUT="${CONNECTOME_OUT:-$HOME/connectome/out}"
VENV="$HOME/connectome/venv"
ONLY="${1:-all}"
# kit:   the Kit runtime of Isaac Sim with the renderer only. Works on a build
#        whose Isaac Sim extensions do not load.
# isaac: the full Isaac Sim application, through SimulationApp.
RENDER="${CONNECTOME_RENDER:-kit}"
mkdir -p "$DATA" "$OUT"

render() {  # package folder
    if [ "$RENDER" = isaac ]; then
        "$ISAACSIM_PATH/python.sh" "$HERE/build_scene.py" "$1" --views 4
    else
        (cd "$ISAACSIM_PATH" && ./kit/kit "$HERE/connectome.viewer.kit" --no-window \
            --ext-folder extscache --ext-folder exts \
            --exec "$HERE/render_kit.py $1 --views 4")
    fi
}

# A small Python environment for the meshing step. Isaac Sim's own Python has
# numpy only once the application is running, and has no h5py.
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet numpy h5py scipy scikit-image pillow
fi

fetch() {  # url, file
    if [ ! -s "$2" ]; then
        curl -L --fail --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
    fi
}

if [ "$ONLY" = all ] || [ "$ONLY" = crop313 ]; then
    fetch https://cellmapchallenge.janelia.org/file/crop313 "$DATA/crop313.glb"
    "$VENV/bin/python" "$HERE/glb_to_package.py" "$DATA/crop313.glb" "$OUT/crop313"
    render "$OUT/crop313"
fi

if [ "$ONLY" = all ] || [ "$ONLY" = cremi_A ]; then
    fetch https://cremi.org/static/data/sample_A_20160501.hdf "$DATA/sample_A_20160501.hdf"
    [ -f "$OUT/cremi_A/manifest.json" ] || \
        "$VENV/bin/python" "$HERE/mesh_labels.py" "$DATA/sample_A_20160501.hdf" "$OUT/cremi_A" --top 40
    render "$OUT/cremi_A"
fi

ls -la "$OUT"/*/render_*.png
