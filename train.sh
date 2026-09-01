device=0

depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_use_DTD_pretrain
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python train.py --dataset DTD --train_data_path /home/L40s/data/DTD \
        --save_path ${save_dir} \
        --features_list 24 --image_size 518  --batch_size 8 --print_freq 1 \
        --epoch 8 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done

# depth=(9)
# n_ctx=(12)
# t_n_ctx=(4)
# for i in "${!depth[@]}";do
#     for j in "${!n_ctx[@]}";do
#     ## train on the VisA dataset
#         base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_use_mvtec_pretrain
#         save_dir=./checkpoints/${base_dir}/
#         CUDA_VISIBLE_DEVICES=${device} python train.py --dataset mvtec --train_data_path /home/L40s/data/mvtec \
#         --save_path ${save_dir} \
#         --features_list 24 --image_size 518  --batch_size 8 --print_freq 1 \
#         --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
#     wait
#     done
# done
