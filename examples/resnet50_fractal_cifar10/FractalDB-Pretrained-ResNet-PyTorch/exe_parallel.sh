
numof_category=1000
fillrate=0.2
weight=0.4
imagesize=362
numof_point=100000
numof_ite=200000
howto_draw='patch_gray'
numof_thread=40
arch=resnet50
data_dir='.' # recommend scratch

# Parameter search
# require cv2 pandas h5py
python param_search/ifs_search.py --rate=${fillrate} --category=${numof_category} --numof_point=${numof_point} --save_dir=${data_dir}'/data'
python param_search/parallel_dir.py --path2dir=${data_dir}'/data' --rate=${fillrate} --category=${numof_category} --thread=${numof_thread}

# Multi-thread processing
for ((i=0 ; i<${numof_thread} ; i++))
do
    python fractal_renderer/make_fractaldb.py \
        --load_root=${data_dir}'/data/csv_rate'${fillrate}'_category'${numof_category}'/csv'${i} \
        --save_root=${data_dir}'/data/FractalDB-'${numof_category} --image_size_x=${imagesize} --image_size_y=${imagesize} \
        --iteration=${numof_ite} --draw_type=${howto_draw} --weight_csv='./fractal_renderer/weights/weights_'${weight}'.csv' &
done
wait

# # FractalDB Pre-training
# require torch torchvision
python pretraining/main.py --path2traindb=${data_dir}'/data/FractalDB-'${numof_category} --dataset='FractalDB-'${numof_category} --numof_classes=${numof_category} --usenet=${arch} --path2weight=${data_dir}'/data/weight'

# # Fine-tuning
python finetuning/main.py --path2db='/data/CIFAR10' --path2weight=${data_dir}'/data/weight' --dataset='FractalDB-'${numof_category} --ft_dataset='CIFAR10' --numof_pretrained_classes=${numof_category} --usenet=${arch}