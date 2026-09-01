device=1

depth=9
n_ctx=12
t_n_ctx=4
epoch=3

dataset=HeadCT
datapath="/home/L40s/data/HeadCT"
# dataset=Br35H
# datapath="/home/L40s/data/Br35H"
# dataset=mpdd
# datapath="/home/L40s/data/MPDD"
# dataset=mvtec
# datapath="/home/L40s/data/mvtec"
# dataset=visa
# datapath="/home/L40s/data/visa"
# dataset=DTD
# datapath="/home/L40s/data/DTD"
# dataset=BrainMRI
# datapath="/home/L40s/data/BrainMRI"
# dataset=SDD
# datapath="/home/L40s/data/SDD"
# dataset=ISBI
# datapath="/home/L40s/data/ISIC"
# dataset=ColonDB
# datapath="/home/L40s/data/CVC-ColonDB"
# dataset=btad
# datapath="/home/L40s/data/BTAD"
# dataset=TN3K
# datapath="/home/L40s/data/TN3K/Thyroid_Dataset/tn3k"


base_dir=${depth}_${n_ctx}_${t_n_ctx}_multiscale_use_DTD_pretrain
save_dir=./checkpoints/${base_dir}/
result_dir=./results/${base_dir}/zero_shot/${dataset}
log_file=${result_dir}/log.txt

mkdir -p ${result_dir}

echo "Config: depth=${depth}, n_ctx=${n_ctx}, t_n_ctx=${t_n_ctx}" >> ${log_file}
echo "---- Testing ${dataset} ----" >> ${log_file}

CUDA_VISIBLE_DEVICES=${device} python test.py \
    --dataset ${dataset} \
    --data_path ${datapath} \
    --save_path ${result_dir} \
    --checkpoint_path ${save_dir}epoch_${epoch}.pth \
    --features_list 24 --image_size 518 \
    --depth ${depth} --n_ctx ${n_ctx} --t_n_ctx ${t_n_ctx} \
    --metrics image-level

echo "" >> ${log_file}