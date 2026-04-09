import h5py
import numpy as np
import os
import scipy.io

# 1. 配置文件路径与切割参数
datasets_config = [
    {
        'src': '/mnt/9944/PDE/NavierStokes_V1e-5_N1200_T20.mat',
        'train_target': './data/large/ns2d_1e-5_train.hdf5',
        'test_target': './data/large/ns2d_1e-5_test.hdf5',
        'train_num': 1000,
        'test_num': 200
    },
    {
        'src': '/mnt/9944/PDE/ns_V1e-3_N5000_T50.mat',
        'train_target': './data/large/ns2d_1e-3_train.hdf5',
        'test_target': './data/large/ns2d_1e-3_test.hdf5',
        # 根据原始代码，5000条通常分 4800(训练) + 200(测试)，
        # 但我们为了和 make_master_file.py 里默认的 1000 对应，这里先切 1000/200 以加速测试。
        # 如果你想全量跑 1e-3，可以改为 4800，并记得同步修改 make_master_file.py。
        'train_num': 1000,  
        'test_num': 200
    }
]

target_dir = './data/large'
os.makedirs(target_dir, exist_ok=True)

for config in datasets_config:
    src_path = config['src']
    print(f"\n🚀 开始处理数据集: {src_path}")
    
    try:
        data = None
        # 尝试 scipy (v7及以下)
        try:
            mat_data = scipy.io.loadmat(src_path)
            data_key = 'u' if 'u' in mat_data else [k for k in mat_data.keys() if not k.startswith('__')][0]
            data = mat_data[data_key].astype(np.float32)
        except Exception:
            # 尝试 h5py (v7.3)
            with h5py.File(src_path, 'r') as f:
                data_key = 'u' if 'u' in f.keys() else list(f.keys())[0]
                data = np.array(f[data_key], dtype=np.float32)

        print(f"✅ 原始维度: {data.shape}")

        # 维度转置逻辑 -> [Batch, X, Y, Time]
        # 处理 1e-5 (可能是 64, 64, 20, 1200)
        if data.ndim == 4 and data.shape[-1] in [1200, 5000]: 
            data = np.transpose(data, (3, 0, 1, 2))
        # 处理 1e-3 (可能是 5000, 64, 64, 50) 或 (50, 64, 64, 5000) 等
        elif data.ndim == 4 and data.shape[0] in [1200, 5000] and data.shape[1] == 64:
            pass # 已经是 [Batch, X, Y, T]，不动
        elif data.ndim == 4 and data.shape[0] in [20, 50] and data.shape[-1] in [1200, 5000]:
            data = np.transpose(data, (3, 1, 2, 0)) # [T, X, Y, Batch] -> [Batch, X, Y, T]
            
        print(f"🔄 转置后维度: {data.shape}")

        # 切割数据
        train_end = config['train_num']
        test_end = train_end + config['test_num']
        
        train_data = data[:train_end]
        test_data = data[train_end:test_end]

        # 保存
        print(f"📦 正在保存...")
        with h5py.File(config['train_target'], 'w') as hf:
            hf.create_dataset('data', data=train_data)
        with h5py.File(config['test_target'], 'w') as hf:
            hf.create_dataset('data', data=test_data)
        print(f"✨ 成功！训练集: {train_data.shape}, 测试集: {test_data.shape}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")