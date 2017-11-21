# Run the simulated visual reach experiment with default hyperparameters
python main.py --experiment=sim_vision_reach --demo_file=data/sim_vision_reach \
                  --demo_gif_dir=data/sim_vision_reach_noisy/ --gif_prefix=color --im_width=80 \
                  --im_height=64 --training_set_size=750 --val_set_size=150 --metatrain_iterations=30000 \
                  --meta_batch_size=25 --train_update_lr=0.001 --clip=True --clip_min=-20 --clip_max=20 --conv_bt=False\
                  --init=xavier --fp=False --num_filters=30 --filter_size=3 --num_conv_layers=3 --num_strides=3 \
                  --num_fc_layers=3 --layer_size=200 --log_dir=logs/sim_vision_reach --gpu_memory_fraction=0.4