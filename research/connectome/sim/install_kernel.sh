#!/bin/bash
# Register a Jupyter kernel that runs the Python of Isaac Sim, so that a notebook
# can import isaacsim (and isaaclab, if installed) in its own process.
#
#   bash sim/install_kernel.sh        # then choose the kernel "Isaac Sim (connectome)"
#
# Nothing is installed into Isaac Sim itself: ipykernel goes into a folder beside
# it (CONNECTOME_KERNEL_PKGS) and is put last on the path of the kernel.
set -euo pipefail
ISAACSIM_PATH="${ISAACSIM_PATH:-$HOME/IsaacSim/_build/linux-aarch64/release}"
PKGS="${CONNECTOME_KERNEL_PKGS:-$HOME/connectome/isaac-kernel-pkgs}"
KDIR="${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/kernels/isaac_sim_connectome"
mkdir -p "$PKGS" "$KDIR"
"$ISAACSIM_PATH/python.sh" -m pip install --quiet --target "$PKGS" ipykernel
cat > "$KDIR/launch.sh" <<LAUNCH
#!/bin/bash
cd "\${CONNECTOME_NOTEBOOK_DIR:-\$PWD}"
exec "$ISAACSIM_PATH/python.sh" -c "import sys; sys.path.append('$PKGS'); from ipykernel import kernelapp; kernelapp.launch_new_instance()" "\$@"
LAUNCH
chmod +x "$KDIR/launch.sh"
cat > "$KDIR/kernel.json" <<JSON
{"argv": ["$KDIR/launch.sh", "-f", "{connection_file}"],
 "display_name": "Isaac Sim (connectome)", "language": "python",
 "env": {"ISAAC_JUPYTER_KERNEL": "1"}}
JSON
echo "kernel installed: $KDIR"
