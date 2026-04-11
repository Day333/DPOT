#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import os
import sys

# 【完美路径修复】绝对可靠地把项目根目录加入环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# 直接导入，不加 try...except，如果有缺依赖包(比如少装了什么库)会直接明明白白地报出来
from data_generation.cfdbench import get_auto_dataset

RAW_DIR = '/mnt/9944/PDE'
SAVE_DIR = '/tmp/data'

def check_exists(paths):
    """检查目标路径是否都已经存在，用于跳过已处理的数据"""
    return all(os.path.exists(p) for p in paths)

def load_mat_robust(file_path):
    """鲁棒地读取 mat 文件，自适应 h5py 和 scipy 的不同读取轴顺序"""
    try:
        # h5py 读取时形状会倒序为 (T, Y, X, N)，需要翻转回 (N, X, Y, T)
        with h5py.File(file_path, 'r') as f:
            return np.array(f['u'], dtype=np.float32).transpose(3, 1, 2, 0)
    except Exception as e_h5:
        try:
            # scipy 读取时直接就是原本的 (N, X, Y, T)，【不要】翻转
            data = scipy.io.loadmat(file_path)['u']
            return np.array(data, dtype=np.float32)
        except Exception as e_scipy:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"\n[致命错误] 无法读取 {file_path}！")
            if size_mb < 1:
                print("--> ⚠️ 强烈警告：文件太小了！可能下载到了 HTML 网页。")
            raise ValueError("文件格式不受支持或已损坏")
            
def process_fno():
    fno_files = {
        'ns2d_fno_1e-3': ('NavierStokes_V1e-3_N5000_T50.mat', 4800, 200),
        'ns2d_fno_1e-4': ('NavierStokes_V1e-4_N10000_T30.mat', 9800, 200),
        'ns2d_fno_1e-5': ('NavierStokes_V1e-5_N1200_T20.mat', 1000, 200)
    }
    for code_id, (fname, n_train, n_test) in fno_files.items():
        train_path = os.path.join(SAVE_DIR, f'{code_id}_train.hdf5')
        test_path = os.path.join(SAVE_DIR, f'{code_id}_test.hdf5')
        if check_exists([train_path, test_path]):
            print(f"[跳过] FNO数据 {code_id} 已处理，跳过。")
            continue
        mat_path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(mat_path):
            continue
        print(f"Processing {fname}...")
        data = load_mat_robust(mat_path)
        train_u, test_u = data[:n_train], data[-n_test:]
        with h5py.File(train_path, 'w') as hf:
            hf.create_dataset('data', data=train_u)
        with h5py.File(test_path, 'w') as hf:
            hf.create_dataset('data', data=test_u)

def process_dr_pdebench():
    save_name = os.path.join(SAVE_DIR, 'dr_pdb')
    train_dir = os.path.join(save_name, 'train')
    test_dir = os.path.join(save_name, 'test')
    if check_exists([train_dir, test_dir]) and len(os.listdir(train_dir)) > 0:
        print("[跳过] dr_pdb 已处理，跳过。")
        return
    raw_path = os.path.join(RAW_DIR, '2D_diff-react_NA_NA.h5')
    n_train, n_test = 900, 100
    if not os.path.exists(raw_path): return
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    data = []
    print(f"Processing 2D_diff-react_NA_NA.h5...")
    with h5py.File(raw_path, 'r') as fp:
        for i in range(len(fp.keys())):
            data.append(fp["{0:0=4d}/data".format(i)])
        data = np.stack(data, axis=0).transpose(0, 2, 3, 1, 4)
    train_ids, test_ids = np.arange(n_train), np.arange(n_train, n_train + n_test)
    for i in tqdm(range(n_train), desc="Saving DR Train"):
        with h5py.File(f'{train_dir}/data_{i}.hdf5', 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
    for i in tqdm(range(n_test), desc="Saving DR Test"):
        with h5py.File(f'{test_dir}/data_{i}.hdf5', 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)

def process_swe_pdebench():
    save_name = os.path.join(SAVE_DIR, 'swe_pdb')
    train_dir = os.path.join(save_name, 'train')
    test_dir = os.path.join(save_name, 'test')
    if check_exists([train_dir, test_dir]) and len(os.listdir(train_dir)) > 0:
        print("[跳过] swe_pdb 已处理，跳过。")
        return
    raw_path = os.path.join(RAW_DIR, '2D_rdb_NA_NA.h5')
    n_train, n_test = 900, 100
    if not os.path.exists(raw_path): return
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    data = []
    print(f"Processing 2D_rdb_NA_NA.h5...")
    with h5py.File(raw_path, 'r') as fp:
        for i in range(len(fp.keys())):
            data.append(fp["{0:0=4d}/data".format(i)])
        data = np.stack(data, axis=0).transpose(0, 2, 3, 1, 4)
    train_ids, test_ids = np.arange(n_train), np.arange(n_train, n_train + n_test)
    for i in tqdm(range(n_train), desc="Saving SWE Train"):
        with h5py.File(f'{train_dir}/data_{i}.hdf5', 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
    for i in tqdm(range(n_test), desc="Saving SWE Test"):
        with h5py.File(f'{test_dir}/data_{i}.hdf5', 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)

def process_ns2d_pdebench(raw_filename, code_id, n_train=9000, n_test=1000):
    raw_path = os.path.join(RAW_DIR, raw_filename)
    save_name = os.path.join(SAVE_DIR, code_id)
    train_dir = os.path.join(save_name, 'train')
    test_dir = os.path.join(save_name, 'test')
    if check_exists([train_dir, test_dir]) and len(os.listdir(train_dir)) > 0:
        print(f"[跳过] PDEBench数据 {code_id} 已处理，跳过。")
        return
    if not os.path.exists(raw_path): return
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Processing {raw_filename}...")
    with h5py.File(raw_path, 'r') as f:
        vx = np.array(f['Vx'], dtype=np.float32)
        vy = np.array(f['Vy'], dtype=np.float32)
        density = np.array(f['density'], dtype=np.float32)
        pressure = np.array(f['pressure'], dtype=np.float32)
        data = np.stack([vx, vy, density, pressure], axis=-1).transpose(0, 2, 3, 1, 4)
    train_ids = np.arange(n_train)
    test_ids = np.arange(n_train, n_train + n_test)
    for i in tqdm(range(n_train), desc=f"Saving {code_id} Train"):
        with h5py.File(os.path.join(train_dir, f'data_{i}.hdf5'), 'w') as f:
            f.create_dataset('data', data=data[train_ids[i]], compression=None)
    for i in tqdm(range(n_test), desc=f"Saving {code_id} Test"):
        with h5py.File(os.path.join(test_dir, f'data_{i}.hdf5'), 'w') as f:
            f.create_dataset('data', data=data[test_ids[i]], compression=None)

def preprocess_pdearena(folder_name, save_folder):
    load_path = os.path.join(RAW_DIR, folder_name)
    save_path = os.path.join(SAVE_DIR, save_folder)
    train_dir = os.path.join(save_path, 'train')
    test_dir = os.path.join(save_path, 'test')
    if check_exists([train_dir, test_dir]) and len(os.listdir(train_dir)) > 0:
        print(f"[跳过] PDEArena数据 {save_folder} 已处理，跳过。")
        return
    if not os.path.exists(load_path): return
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    test_tot, train_tot = 0, 0
    for root, _, files in os.walk(load_path):
        for file in tqdm(files, desc=f"Processing {folder_name}"):
            if not file.endswith('.h5'): continue
            try:
                with h5py.File(os.path.join(root, file), 'r') as f:
                    if 'test' in file: internal_key, target_dir = 'test', test_dir
                    elif 'train' in file: internal_key, target_dir = 'train', train_dir
                    elif 'valid' in file: internal_key, target_dir = 'valid', train_dir
                    else: continue
                    u, vx, vy = f[internal_key]['u'][:], f[internal_key]['vx'][:], f[internal_key]['vy'][:]
                    out = np.stack([u, vx, vy], axis=-1).transpose(0, 2, 3, 1, 4)
                    for data in out:
                        idx = test_tot if internal_key == 'test' else train_tot
                        if internal_key == 'test': test_tot += 1
                        else: train_tot += 1
                        with h5py.File(os.path.join(target_dir, f'data_{idx}.hdf5'), 'w') as g:
                            g.create_dataset('data', data=data)
            except Exception as e:
                print(f'Error in file {file}: {e}')

def preprocess_cfdbench():
    """处理 CFDBench 数据集"""
    save_dir = os.path.join(SAVE_DIR, 'cfdbench')
    train_path = os.path.join(save_dir, 'ns2d_cdb_train.hdf5')
    test_path = os.path.join(save_dir, 'ns2d_cdb_test.hdf5')

    if check_exists([train_path, test_path]):
        print("[跳过] CFDBench 已处理，跳过。")
        return

    raw_cfdbench_dir = os.path.join(RAW_DIR, 'CFDBench')
    if not os.path.exists(raw_cfdbench_dir):
        print(f"[跳过] 找不到 CFDBench 原始数据目录: {raw_cfdbench_dir}，如果你还没下载请忽略。")
        return

    os.makedirs(save_dir, exist_ok=True)
    print("Processing CFDBench...")

    # CFDBench 原始代码依赖于传递 Path 对象
    train_data_cavity, dev_data_cavity, test_data_cavity = get_auto_dataset(
        data_dir=Path(raw_cfdbench_dir), data_name='cavity_prop_bc_geo', delta_time=0.1, norm_props=True, norm_bc=True
    )
    train_data_cylinder, dev_data_cylinder, test_data_cylinder = get_auto_dataset(
        data_dir=Path(raw_cfdbench_dir), data_name='cylinder_prop_bc_geo', delta_time=0.1, norm_props=True, norm_bc=True
    )
    train_data_tube, dev_data_tube, test_data_tube = get_auto_dataset(
        data_dir=Path(raw_cfdbench_dir), data_name='tube_prop_bc_geo', delta_time=0.1, norm_props=True, norm_bc=True
    )

    train_feats = train_data_cavity.all_features + train_data_cylinder.all_features + train_data_tube.all_features
    test_feats = test_data_cavity.all_features + test_data_cylinder.all_features + test_data_tube.all_features

    infer_steps = 20

    def split_trajectory(data_list, time_step, grid_size=64):
        traj_split = []
        for i, x in enumerate(data_list):
            T = x.shape[0]
            num_segments = int(np.ceil(T / time_step))
            padded_length = num_segments * time_step
            padded_array = np.zeros((padded_length, *x.shape[1:]))
            padded_array[:T, ...] = x
            if T % time_step != 0:
                last_frame = x[-1, ...]
                padded_array[T:, ...] = last_frame
            # 使用 PyTorch 进行空间插值
            padded_array = F.interpolate(torch.from_numpy(padded_array), size=(grid_size, grid_size), mode='bilinear', align_corners=True).numpy()
            padded_array = padded_array.reshape((num_segments, time_step, *padded_array.shape[1:]))
            traj_split.append(padded_array)
        return np.concatenate(traj_split, axis=0)

    print("Splitting CFDBench trajectories and applying interpolation...")
    train_data = split_trajectory(train_feats, infer_steps, grid_size=64)
    test_data = split_trajectory(test_feats, infer_steps, grid_size=64)
    
    # 按照 [B, X, Y, T, C] 进行转置
    train_data = train_data.transpose(0, 3, 4, 1, 2)
    test_data = test_data.transpose(0, 3, 4, 1, 2)
    
    print(f"CFDBench shapes - Train: {train_data.shape}, Test: {test_data.shape}")

    with h5py.File(train_path, 'w') as fp:
        fp.create_dataset('data', data=train_data, compression=None)

    with h5py.File(test_path, 'w') as fp:
        fp.create_dataset('data', data=test_data, compression=None)


if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Starting data preprocessing...")
    
    process_fno()
    process_dr_pdebench()
    process_swe_pdebench()
    process_ns2d_pdebench('2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5', 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2')
    process_ns2d_pdebench('2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5', 'ns2d_pdb_M1_eta1e-2_zeta1e-2')
    preprocess_pdearena('NavierStokes-2D', 'ns2d_pda')
    preprocess_pdearena('NavierStokes-2D-conditoned', 'ns2d_cond_pda')
    
    preprocess_cfdbench()
    
    print(f"\n✅ 数据处理检查完毕！文件存储在 {SAVE_DIR}")