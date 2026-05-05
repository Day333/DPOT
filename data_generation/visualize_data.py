#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.ndimage

def visualize_all_channels(base_dir, dataset_name, data_id=0, target_t=0):
    path = os.path.join(base_dir, dataset_name)
    
    try:
        if os.path.isdir(os.path.join(path, 'test')):
            file_path = os.path.join(path, 'test', f'data_{data_id}.hdf5')
            print(f"Loading Scatter Data: {file_path}")
            with h5py.File(file_path, 'r') as f:
                raw_data = f['data'][:] 
        elif os.path.isfile(path + '_test.hdf5'):
            file_path = path + '_test.hdf5'
            print(f"Loading Packed Data: {file_path}")
            with h5py.File(file_path, 'r') as f:
                raw_data = f['data'][data_id] 
        elif os.path.isfile(os.path.join(path, 'ns2d_cdb_test.hdf5')):
            file_path = os.path.join(path, 'ns2d_cdb_test.hdf5')
            print(f"Loading CFDBench Data: {file_path}")
            with h5py.File(file_path, 'r') as f:
                raw_data = f['data'][data_id]
        else:
            print(f"[Warning] Skipping '{dataset_name}': No valid path found.\n")
            return

        original_shape = raw_data.shape

        if len(raw_data.shape) == 4:
            num_channels = raw_data.shape[-1]
            has_channel_dim = True
        else:
            num_channels = 1
            has_channel_dim = False
            
        max_t = raw_data.shape[2]
        safe_t = target_t if target_t < max_t else max_t - 1

        print(f"Dataset: {dataset_name}")
        print(f"Original Shape: {original_shape} | Channels: {num_channels} | Requested T: {target_t} | Target T: {safe_t}")

        fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))
        if num_channels == 1:
            axes = [axes]
            
        for c in range(num_channels):
            x = raw_data[:, :, safe_t, c] if has_channel_dim else raw_data[:, :, safe_t]
            
            if x.shape[0] < 128:
                x = scipy.ndimage.zoom(x, (128/x.shape[0], 128/x.shape[1]), order=3)

            ax = axes[c]
            im = ax.imshow(x, cmap='plasma', origin='lower')
            ax.set_title(f'Ch {c}')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) 

        plt.suptitle(f"Dataset: {dataset_name} | ID: {data_id} | T: {safe_t}", fontsize=12, y=1.02)
        
        save_name = f'vis_{dataset_name}_id{data_id}_t{safe_t}_allC.png'
        plt.savefig(save_name, bbox_inches='tight', dpi=150)
        plt.close() 
        print(f"[Success] Saved visualization to: {save_name}\n")

    except Exception as e:
        print(f"[Error] Failed to process '{dataset_name}': {e}\n")


if __name__ == '__main__':
    BASE_DIR = '/tmp/data'
    
    datasets = [
        'ns2d_fno_1e-5',
        'ns2d_fno_1e-4',
        'ns2d_fno_1e-3',
        'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
        'ns2d_pdb_M1_eta1e-2_zeta1e-2',
        'swe_pdb',
        'dr_pdb',
        'cfdbench',
        'ns2d_pda',
        'ns2d_cond_pda'
    ]
    
    SAMPLE_ID = 0   
    TIME_STEP = 5   

    print(f"Starting batch visualization...\nBase Directory: {BASE_DIR}\n" + "-"*60)
    
    for ds in datasets:
        visualize_all_channels(BASE_DIR, ds, data_id=SAMPLE_ID, target_t=TIME_STEP)

    print("-" * 60 + "\nAll datasets have been processed successfully!")