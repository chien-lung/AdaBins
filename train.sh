#!/bin/sh

#SBATCH --job-name="Train modification of Adabins"
#SBATCH --mail-type=BEGIN,END
#SBATCH --time=24:00:00
#SBATCH --account=eecs545w21_class

#SBATCH --mem=7gb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1


module load python
module load cuda
module load cudnn
source ../depth_estimation/bin/activate

python train.py --epochs 20 \
                --dataset nyu \
                --data_path dataset/sync/ \
                --gt_path dataset/sync/ \
                --filenames_file train_test_inputs/nyudepthv2_train_files_with_gt_11000.txt \
                --filenames_file_eval train_test_inputs/nyudepthv2_test_files_with_gt_200.txt \
                --data_path_eval dataset/official_splits/test/ \
                --gt_path_eval dataset/official_splits/test/ \
                --bs 4 \
                --wd 0.0001 \
                --lr 0.0001 \
                --workers 1 \
                --validate_every 500 \
                --w_grad 1.0 \
                --w_norm 1.0 \
