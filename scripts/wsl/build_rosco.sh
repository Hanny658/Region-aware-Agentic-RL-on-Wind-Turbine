#!/bin/bash
# Build the patched ROSCO controller (libdiscon.so with ZeroMQ, 22-channel measurements) inside WSL.
# Prereq: scripts/wsl/setup_env.sh has run (conda env "wtrl" provides gfortran/cmake/zeromq).
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
WTRL=~/wtrl
PROJ=/mnt/c/Users/hanny/Desktop/MyProjectSpace/Region-aware-Agentic-RL-on-Wind-Turbine
export MAMBA_ROOT_PREFIX="$WTRL/mamba"
eval "$("$WTRL/bin/micromamba" shell hook -s bash)"
micromamba activate wtrl

SRC="$WTRL/ROSCO/rosco/controller"
# apply patch (keep pristine copies for diffing)
for f in ZeroMQInterface.f90 zmq_client.c Controllers.f90 Filters.f90 ControllerBlocks.f90; do
  [ -f "$SRC/src/$f.orig" ] || cp "$SRC/src/$f" "$SRC/src/$f.orig"
  tr -d '\r' < "$PROJ/controllers/rosco_patch/$f" > "$SRC/src/$f"
done
diff -u "$SRC/src/ZeroMQInterface.f90.orig" "$SRC/src/ZeroMQInterface.f90" | head -5 || true

BUILD="$WTRL/rosco_build"
rm -rf "$BUILD" && mkdir -p "$BUILD"
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$WTRL/rosco_install" 2>&1 | tee "$BUILD/cmake.log" | grep -i "zmq\|zeromq\|Fortran\|error" || true
grep -q "ZMQ_CLIENT" "$BUILD/CMakeCache.txt" 2>/dev/null && echo "ZMQ define present in cache" || true
cmake --build "$BUILD" -j8 2>&1 | tail -3
cmake --install "$BUILD" 2>&1 | tail -2
ls -la "$WTRL/rosco_install/lib/"
# verify the ZMQ symbol actually got linked in
nm -D "$WTRL/rosco_install/lib/libdiscon.so" | grep -i "zmq_client\|zmq_send" | head -3
ldd "$WTRL/rosco_install/lib/libdiscon.so" | grep -i "zmq\|gfortran" || true
echo "BUILD_DONE"
