### script for writing meta information of datasets into master.csv
### for node property prediction datasets.
import os

DATASET_DICT = {}
DATASET_LIST = []

BASE_DATA_DIR = '/tmp/data'

# =====================================================================
# 1. FNO Datasets 
# =====================================================================
name = 'ns2d_fno_1e-5'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, 'ns2d_fno_1e-5_train.hdf5'), 
    'test_path': os.path.join(BASE_DATA_DIR, 'ns2d_fno_1e-5_test.hdf5')
}
DATASET_DICT[name]['train_size'] = 1000
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage']= False
DATASET_DICT[name]['t_test'] = 10
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 20
DATASET_DICT[name]['in_size'] = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'ns2d_fno_1e-4'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, 'ns2d_fno_1e-4_train.hdf5'), 
    'test_path': os.path.join(BASE_DATA_DIR, 'ns2d_fno_1e-4_test.hdf5')
}
DATASET_DICT[name]['train_size'] = 9800
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['t_test'] = 20
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 30
DATASET_DICT[name]['in_size'] = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'ns2d_fno_1e-3'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, 'ns2d_fno_1e-3_train.hdf5'), 
    'test_path': os.path.join(BASE_DATA_DIR, 'ns2d_fno_1e-3_test.hdf5')
}
DATASET_DICT[name]['train_size'] = 1000
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['t_test'] = 20
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 50
DATASET_DICT[name]['in_size'] = (64, 64)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['downsample'] = (1, 1)

# =====================================================================
# 2. PDEBench Datasets
# =====================================================================
name = 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, name, 'train'), 
    'test_path': os.path.join(BASE_DATA_DIR, name, 'test')
}
DATASET_DICT[name]['train_size'] = 9000
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'ns2d_pdb_M1_eta1e-2_zeta1e-2'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, name, 'train'), 
    'test_path': os.path.join(BASE_DATA_DIR, name, 'test')
}
DATASET_DICT[name]['train_size'] = 9000
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 11
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 21
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 4
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'swe_pdb'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, 'swe_pdb', 'train'), 
    'test_path': os.path.join(BASE_DATA_DIR, 'swe_pdb', 'test')
}
DATASET_DICT[name]['train_size'] = 900
DATASET_DICT[name]['test_size'] = 60
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 91
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 101
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 1
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'dr_pdb'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, 'dr_pdb', 'train'), 
    'test_path': os.path.join(BASE_DATA_DIR, 'dr_pdb', 'test')
}
DATASET_DICT[name]['train_size'] = 900
DATASET_DICT[name]['test_size'] = 60
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 91
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 101
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 2
DATASET_DICT[name]['downsample'] = (1, 1)

# =====================================================================
# 3. CFDBench Datasets
# =====================================================================
name = 'cfdbench'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, 'cfdbench', 'ns2d_cdb_train.hdf5'), 
    'test_path': os.path.join(BASE_DATA_DIR, 'cfdbench', 'ns2d_cdb_test.hdf5')
}
DATASET_DICT[name]['train_size'] = 9000
DATASET_DICT[name]['test_size'] = 1000
DATASET_DICT[name]['scatter_storage'] = False
DATASET_DICT[name]['t_test'] = 20
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 20
DATASET_DICT[name]['in_size'] = (64, 64)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['pred_channels'] = 2
DATASET_DICT[name]['downsample'] = (1, 1)

# =====================================================================
# 4. PDEArena Datasets (散点存储: True)
# =====================================================================
name = 'ns2d_pda'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, 'ns2d_pda', 'train'), 
    'test_path': os.path.join(BASE_DATA_DIR, 'ns2d_pda', 'test')
}
DATASET_DICT[name]['train_size'] = 6500
DATASET_DICT[name]['test_size'] = 650
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 4
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 14
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['downsample'] = (1, 1)

name = 'ns2d_cond_pda'
DATASET_DICT[name] = {
    'train_path': os.path.join(BASE_DATA_DIR, 'ns2d_cond_pda', 'train'), 
    'test_path': os.path.join(BASE_DATA_DIR, 'ns2d_cond_pda', 'test')
}
DATASET_DICT[name]['train_size'] = 3100
DATASET_DICT[name]['test_size'] = 200
DATASET_DICT[name]['scatter_storage'] = True
DATASET_DICT[name]['t_test'] = 46
DATASET_DICT[name]['t_in'] = 10
DATASET_DICT[name]['t_total'] = 56
DATASET_DICT[name]['in_size'] = (128, 128)
DATASET_DICT[name]['n_channels'] = 3
DATASET_DICT[name]['downsample'] = (1, 1)