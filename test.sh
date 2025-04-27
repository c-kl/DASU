#!/bin/bash
export model=GASA_e400_s8
export ck=best
echo ${model}
#Please change '--test' to '--save' to save the images

#echo 'Middleburry'
echo '4'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py  --test --checkpoint ${ck}  --name ${model} --model GASA  --dataset Middlebury --scale 4 --data_root ./data/depth_enhance/06_Middlebury_Dataset
echo '8'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Middlebury --scale 8 --data_root ./data/depth_enhance/06_Middlebury_Dataset
echo '16'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Middlebury --scale 16 --data_root ./data/depth_enhance/06_Middlebury_Dataset
echo '3.75'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py  --test --checkpoint ${ck}  --name ${model} --model GASA  --dataset Middlebury --scale 3.75 --data_root ./data/depth_enhance/06_Middlebury_Dataset
echo '14.6'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Middlebury --scale 14.6 --data_root ./data/depth_enhance/06_Middlebury_Dataset
echo '17.05'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py  \
--test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Middlebury --scale 17.05 --data_root ./data/depth_enhance/06_Middlebury_Dataset

echo 'RGBD'
echo '3.75'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 3.75 --data_root ./data/depth_enhance/03_RGBD_Dataset
echo '14.6'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 14.6 --data_root ./data/depth_enhance/03_RGBD_Dataset
echo '17.05'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py \
--test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 17.05 --data_root ./data/depth_enhance/03_RGBD_Dataset
echo '4'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 4 --data_root ./data/depth_enhance/03_RGBD_Dataset
echo '8'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 8 --data_root ./data/depth_enhance/03_RGBD_Dataset
echo '16'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset Lu --scale 16 --data_root ./data/depth_enhance/03_RGBD_Dataset

##
echo 'NYU'
echo '3.75'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 3.75
echo '14.6'
OMP_NUM_THREADS=20.25 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 14.6
echo '17.05'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py \
--test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 17.05
echo '4'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 4
cho '8'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py  --test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 8
echo '16'
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main.py \
--test --checkpoint ${ck}   --name ${model} --model GASA  --dataset NYU  --scale 16


#

