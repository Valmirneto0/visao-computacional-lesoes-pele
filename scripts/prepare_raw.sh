#!/usr/bin/env bash
# Descompacta os zips do ISIC 2018 em data/raw/ (idempotente).
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
for z in ISIC2018_Task1-2_Training_Input ISIC2018_Task1_Training_GroundTruth ISIC2018_Task3_Training_Input ISIC2018_Task3_Training_GroundTruth; do
  if [ -f "$z.zip" ] && [ ! -d "$z" ]; then
    echo "descompactando $z.zip"; unzip -q "$z.zip" -d .
  fi
done
ls -1
