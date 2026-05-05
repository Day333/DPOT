python train_temporal_non.py \
    --model DPOT \
    --train_paths swe_pdb \
    --test_paths swe_pdb \
    --batch_size 32 \
    --epochs 100 \
    --warmup_epochs 10 \
    --n_layers 4 \
    --n_blocks 4 \
    --mlp_ratio 1 \
    --ntrain_list 900 \
    --T_bundle 10 \
    --T_ar 10 \
    --width 512 \
    --gpu 3

# --use_writer \
# train_paths: [
#   'ns2d_fno_1e-5',
#   'ns2d_fno_1e-3',
#   'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
#   'swe_pdb',
#   'dr_pdb',
#   'cfdbench'
# ]