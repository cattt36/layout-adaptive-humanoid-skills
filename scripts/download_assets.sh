#!/bin/bash
# download_assets.sh <target dir>: downloads the release assets (models, processed data, results) of v1.0 and unpacks
# them into <target dir> (-> <target dir>/models, data, runs).  Needs curl and about 2 x 11 GB of free space.
set -e
T=${1:?usage: download_assets.sh <target dir>}
URL=https://github.com/cattt36/layout-adaptive-humanoid-skills/releases/download/v1.0
ASSETS="models_experts_part1 models_skill_encoders_part1 models_students_part1 models_students_part2
        models_deploy_exports_part1 data_part1 data_part2 results_part1"
mkdir -p "$T"
cd "$T"
curl -fL -O "$URL/SHA256SUMS"
for a in $ASSETS; do
  [ -f "$a.tar" ] || curl -fL -C - -O "$URL/$a.tar"
  grep " $a.tar\$" SHA256SUMS | sha256sum -c -
done
for a in $ASSETS; do tar xf "$a.tar" && echo "unpacked $a"; done
echo "done; the .tar files can be deleted"
