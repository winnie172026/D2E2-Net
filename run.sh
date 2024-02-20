#!/usr/bin/env bash


ratio='0.10'
#dataset='TN'
dataset='TN'
for repeat in 1 2
do
# detection
cd code_detection || exit
echo ${PWD}

/home/kunzixie/Medical_Image_Analysis/venv/bin/python main.py --random-seed -1 --lr 0.0001 --batch-size 16 --epochs 80 \
  --gpus 0 --root-save-dir ../experiments/detection/${dataset}/${ratio}_repeat=${repeat}

# segmentation
cd ../code_seg || exit
echo ${PWD}
/home/kunzixie/Medical_Image_Analysis/venv/bin/python main.py --random-seed -1 --lr 0.0001 --batch-size 8 --epochs 120 \
  --gpus 0 --save-dir ../experiments/segmentation/${dataset}/${ratio}_repeat=${repeat} \
  --detection-results-dir ../experiments/detection/${dataset}/${ratio}_repeat=${repeat}/3/best/images_prob_maps
done