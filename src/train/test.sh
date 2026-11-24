SIF=/lustre1/work/c30897/wtq/env/singularity/axolotl-ngc2605.sif

echo "===== System ====="
uname -r

echo "===== Singularity ====="
singularity --version

echo "===== Image ====="
ls -lh $SIF
file $SIF

echo "===== SIF structure ====="
singularity sif list $SIF

echo "===== SquashFS ====="
unsquashfs -s $SIF
