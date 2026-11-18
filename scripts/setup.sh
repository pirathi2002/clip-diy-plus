#!/usr/bin/env bash
# Fetch third-party code and PASCAL VOC 2012. COCO is optional: pass --coco.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p third_party data
[ -d third_party/clip-diy ] || git clone --depth 1 https://github.com/wysoczanska/clip-diy.git third_party/clip-diy
[ -d third_party/FOUND ] || git clone --depth 1 https://github.com/valeoai/FOUND.git third_party/FOUND
[ -d third_party/dino ] || git clone --depth 1 https://github.com/facebookresearch/dino.git third_party/dino

if [ ! -d data/VOCdevkit/VOC2012 ]; then
  # pjreddie mirror is much faster than host.robots.ox.ac.uk; md5 is checked below
  curl -L -C - --retry 10 -o data/VOCtrainval_11-May-2012.tar https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
  echo "6cd6e144f989b92b3379bac3b3de84fd  data/VOCtrainval_11-May-2012.tar" | md5sum -c -
  tar xf data/VOCtrainval_11-May-2012.tar -C data && rm data/VOCtrainval_11-May-2012.tar
fi

if [[ "${1:-}" == "--coco" && ! -d data/coco/val2017 ]]; then
  mkdir -p data/coco/annotations
  curl -L -C - --retry 10 -o data/coco/val2017.zip http://images.cocodataset.org/zips/val2017.zip
  curl -L -C - --retry 10 -o data/coco/ann.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q data/coco/val2017.zip -d data/coco
  unzip -q -j data/coco/ann.zip annotations/instances_val2017.json -d data/coco/annotations
  rm data/coco/val2017.zip data/coco/ann.zip
fi
echo "setup done"
