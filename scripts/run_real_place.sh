# Run the real place experiment with default hyperparameters
python main.py --experiment=real_place --demo_file=/raid/kevin/maml_place \
                  --demo_gif_dir=/raid/kevin/maml_place/demo_gifs/ --gif_prefix=object --T=30 \
                  --im_width=100 --im_height=90 --val_set_size=12 --metatrain_iterations=50000 \
                  --meta_batch_size=12 --train_update_lr=0.005  --num_updates=5 --clip=True --clip_min=-30 \
                  --clip_max=30 --fp=True --num_filters=64 --filter_size=3 --num_conv_layers=5 --num_strides=3 \
                  --bt_dim=20 --all_fc_bt=True --num_fc_layers=3 --layer_size=100 --loss_multiplier=100.0  \
                  --conv_bt=False --two_head=False --learn_final_eept=True --use_l1_l2_loss=True  \
                  --pretrain_weight_path=data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 \
                  --log_dir=/raid/kevin/real_place