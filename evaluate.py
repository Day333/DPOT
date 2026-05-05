import sys
import os
sys.path.append(['.','./../'])
os.environ['OMP_NUM_THREADS'] = '16'

import json
import time
import argparse
import torch
import numpy as np

from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from utils.optimizer import Adam, Lamb
from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint
from utils.criterion import SimpleLpLoss
from utils.griddataset import MixedTemporalDataset
from models.fno import FNO2d
from models.dpot import DPOTNet

# torch.manual_seed(0)
# np.random.seed(0)



################################################################
# configs
################################################################


parser = argparse.ArgumentParser(description='Training or pretraining for the same data type')

### currently no influence
parser.add_argument('--model', type=str, default='AFNO')
parser.add_argument('--dataset',type=str, default='ns2d')

parser.add_argument('--train_paths',nargs='+', type=str, default=[
 'ns2d_fno_1e-5',
  'ns2d_fno_1e-3',
  'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
  'swe_pdb',
  'dr_pdb',
  'cfdbench'
])
parser.add_argument('--test_paths',nargs='+',type=str, default=[
 'ns2d_fno_1e-5',
  'ns2d_fno_1e-3',
  'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2',
  'swe_pdb',
  'dr_pdb',
  'cfdbench'
])
parser.add_argument('--resume_path',type=str, default='/root/files/pdessl/logs_pretrain/AFNO_ns2d_1218_17_20_14:S_12_114400/model_99.pth')
parser.add_argument('--ntrain_list', nargs='+', type=int, default=None)
parser.add_argument('--ntest_list',nargs='+',type=int, default=None)
parser.add_argument('--data_weights',nargs='+',type=int, default=[1])
parser.add_argument('--use_writer', action='store_true',default=False)

parser.add_argument('--res', type=int, default=128)
parser.add_argument('--noise_scale',type=float, default=0.0)
# parser.add_argument('--n_channels',type=int,default=-1)

### shared params
parser.add_argument('--width', type=int, default=1024)
parser.add_argument('--n_layers',type=int, default=6)
parser.add_argument('--act',type=str, default='gelu')

### GNOT params
parser.add_argument('--max_nodes',type=int, default=-1)

### FNO params
parser.add_argument('--modes', type=int, default=20)
parser.add_argument('--use_ln',type=int, default=0)
parser.add_argument('--normalize',type=int, default=0)


### AFNO
parser.add_argument('--patch_size',type=int, default=8)
parser.add_argument('--n_blocks',type=int, default=8)
parser.add_argument('--mlp_ratio',type=int, default=1)
parser.add_argument('--out_layer_dim', type=int, default=32)


parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
parser.add_argument('--beta1',type=float,default=0.9)
parser.add_argument('--beta2',type=float,default=0.999)
parser.add_argument('--lr_method',type=str, default='step')
parser.add_argument('--grad_clip',type=float, default=10000.0)
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--step_gamma', type=float, default=0.5)
parser.add_argument('--warmup_epochs',type=int, default=50)
parser.add_argument('--sub', type=int, default=1)
parser.add_argument('--T_in', type=int, default=10)
parser.add_argument('--T_ar', type=int, default=1)
parser.add_argument('--T_bundle', type=int, default=1)
parser.add_argument('--gpu', type=str, default="3")
parser.add_argument('--comment',type=str, default="")
parser.add_argument('--log_path',type=str,default='')


parser.add_argument('--n_channels',type=int, default=4)
parser.add_argument('--n_class',type=int,default=12)

args = parser.parse_args()


device = torch.device("cuda:{}".format(args.gpu))

print(f"Current working directory: {os.getcwd()}")




################################################################
# load data and dataloader
################################################################
train_paths = args.train_paths
test_paths = args.test_paths
args.data_weights = [1] * len(args.train_paths) if len(args.data_weights) == 1 else args.data_weights
print('args',args)


train_dataset = MixedTemporalDataset(args.train_paths, args.ntrain_list, res=args.res, t_in = args.T_in, t_ar = args.T_ar, normalize=False,train=True, data_weights=args.data_weights, n_channels=args.n_channels)
test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=-1, normalize=False, train=False) for i, test_path in enumerate(test_paths)]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_dataset in test_datasets]
ntrain, ntests = len(train_dataset), [len(test_dataset) for test_dataset in test_datasets]
print('Train num {} test num {}'.format(train_dataset.n_sizes, ntests))
################################################################
# load model
################################################################
if args.model == "FNO":
    model = FNO2d(args.modes, args.modes, args.width, img_size = args.res, patch_size=args.patch_size, in_timesteps = args.T_in, out_timesteps=1,normalize=args.normalize,n_layers = args.n_layers,use_ln = args.use_ln, n_channels=train_dataset.n_channels, n_cls=len(args.train_paths)).to(device)
elif args.model == 'DPOT':
    model = DPOTNet(img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels, in_timesteps = args.T_in, out_timesteps = args.T_bundle, out_channels=train_dataset.n_channels, normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers, n_blocks = args.n_blocks, mlp_ratio=args.mlp_ratio, out_layer_dim=args.out_layer_dim, act=args.act, n_cls=args.n_class).to(device)
else:
    raise NotImplementedError

if args.resume_path:
    print('Loading models and fine tune from {}'.format(args.resume_path))
    args.resume_path = args.resume_path

    load_model_from_checkpoint(model, torch.load(args.resume_path,map_location='cuda:{}'.format(args.gpu))['model'])

#### set optimizer
if args.opt == 'lamb':
    optimizer = Lamb(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2), adam=True, debias=False,weight_decay=1e-4)
else:
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)


if args.lr_method == 'cycle':
    print('Using cycle learning rate schedule')
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, div_factor=1e4, pct_start=(args.warmup_epochs / args.epochs), final_div_factor=1e4, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.lr_method == 'step':
    print('Using step learning rate schedule')
    scheduler = StepLR(optimizer, step_size=args.step_size * len(train_loader), gamma=args.step_gamma)
elif args.lr_method == 'warmup':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: min((steps + 1) / (args.warmup_epochs * len(train_loader)), np.power(args.warmup_epochs * len(train_loader) / float(steps + 1), 0.5)))
elif args.lr_method == 'linear':
    print('Using warmup learning rate schedule')
    scheduler = LambdaLR(optimizer, lambda steps: (1 - steps / (args.epochs * len(train_loader))))
elif args.lr_method == 'restart':
    print('Using cos anneal restart')
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader) * args.lr_step_size, eta_min=0.)
elif args.lr_method == 'cyclic':
    scheduler = CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=args.lr_step_size * len(train_loader),mode='triangular2', cycle_momentum=False)
else:
    raise NotImplementedError

comment = args.comment + '_{}_{}'.format(len(train_paths), ntrain)
log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
model_path = log_path + '/model.pth'
if args.use_writer:
    writer = SummaryWriter(log_dir=log_path)
    fp = open(log_path + '/logs.txt', 'w+',buffering=1)
    json.dump(vars(args), open(log_path + '/params.json', 'w'),indent=4)
    sys.stdout = fp

else:
    writer = None
print(model)
count_parameters(model)

################################################################
# Main function for pretraining
################################################################
myloss = SimpleLpLoss(size_average=False)
clsloss = torch.nn.CrossEntropyLoss(reduction='sum')


import time

test_l2_fulls, test_l2_steps = [], []
time_test, total_samples = 0., 0

model.eval()
with torch.no_grad():
    for id, test_loader in enumerate(test_loaders):
        test_l2_full, test_l2_step = 0, 0
        for xx, yy, msk, _ in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            msk = msk.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            time_i = time.time()

            for t in range(0, yy.shape[-2], args.T_bundle):
                y = yy[..., t:t + args.T_bundle, :]

                im, _ = model(xx)

                loss += myloss(im, y, mask=msk)

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -2)

                xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            time_test += time.time() - time_i

            total_samples += xx.shape[0]

            test_l2_step += loss.item()
            test_l2_full += myloss(pred, yy, mask=msk)

        test_l2_step_avg = test_l2_step / ntests[id] / (yy.shape[-2] / args.T_bundle)
        test_l2_full_avg = test_l2_full / ntests[id]
        test_l2_steps.append(test_l2_step_avg)
        test_l2_fulls.append(test_l2_full_avg.item())

print(test_l2_fulls)
for i in range(len(test_paths)):
    print('{}: {:.5f}'.format(test_paths[i], test_l2_fulls[i]))

print('Total rollout time: {:.6f}s'.format(time_test))
print('Total samples: {}'.format(int(total_samples)))
print('Avg rollout time per sample: {:.6f}s'.format(time_test / total_samples))

# import sys
# import os
# sys.path.append(['.','./../'])
# os.environ['OMP_NUM_THREADS'] = '16'

# import json
# import time
# import argparse
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.ndimage

# from torch.optim.lr_scheduler import OneCycleLR, StepLR, LambdaLR, CosineAnnealingWarmRestarts, CyclicLR
# from torch.utils.tensorboard import SummaryWriter
# from utils.optimizer import Adam, Lamb
# from utils.utilities import count_parameters, get_grid, load_model_from_checkpoint
# from utils.criterion import SimpleLpLoss
# from utils.griddataset import MixedTemporalDataset
# from models.fno import FNO2d
# from models.dpot import DPOTNet

# # torch.manual_seed(0)
# # np.random.seed(0)

# ################################################################
# # Visualization Helper
# ################################################################
# def visualize_all_timesteps(true_data, pred_data, dataset_name, sample_idx=0, save_dir='./visualizations'):
#     """
#     遍历整个时间维度，为每个时间步生成多通道的 Ground Truth 与预测值对比图。
#     假定输入 shape 为 (B, H, W, T, C)。
#     """
#     os.makedirs(save_dir, exist_ok=True)

#     # 移动到 CPU 并转为 numpy
#     true_np = true_data[sample_idx].cpu().numpy()
#     pred_np = pred_data[sample_idx].cpu().numpy()

#     if len(true_np.shape) != 4:
#         print(f"[Warning] 期望的数据形状为 4D (H, W, T, C)，但得到 {true_np.shape}。跳过可视化。")
#         return

#     # 时间步长 T 在索引 2
#     T = true_np.shape[2]
#     num_channels = true_np.shape[-1]

#     # 遍历每一个时间步
#     for t in range(T):
#         fig, axes = plt.subplots(2, num_channels, figsize=(5 * num_channels, 8))
        
#         # 确保 axes 是 2D 数组（兼容单通道的情况）
#         if num_channels == 1:
#             axes = np.expand_dims(axes, axis=1)

#         for c in range(num_channels):
#             # --- 绘制 Ground Truth ---
#             x_true = true_np[:, :, t, c]
#             # 保持你原来的插值放大逻辑
#             if x_true.shape[0] < 128:
#                 x_true = scipy.ndimage.zoom(x_true, (128/x_true.shape[0], 128/x_true.shape[1]), order=3)
            
#             ax_true = axes[0, c]
#             im_true = ax_true.imshow(x_true, cmap='plasma', origin='lower')
#             ax_true.set_title(f'True - Ch {c}')
#             ax_true.axis('off')
#             fig.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)

#             # --- 绘制 Prediction ---
#             x_pred = pred_np[:, :, t, c]
#             if x_pred.shape[0] < 128:
#                 x_pred = scipy.ndimage.zoom(x_pred, (128/x_pred.shape[0], 128/x_pred.shape[1]), order=3)
                
#             ax_pred = axes[1, c]
#             im_pred = ax_pred.imshow(x_pred, cmap='plasma', origin='lower')
#             ax_pred.set_title(f'Pred - Ch {c}')
#             ax_pred.axis('off')
#             fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

#         # 标题显示当前的时间步进度
#         plt.suptitle(f"Dataset: {dataset_name} | Timestep: {t+1} / {T}", fontsize=14, y=1.02)
        
#         # 文件名带上前导零以保证在文件夹里按顺序排列 (例如 t000.png, t001.png)
#         save_name = os.path.join(save_dir, f'vis_{dataset_name}_t{t:03d}.png')
#         plt.savefig(save_name, bbox_inches='tight', dpi=150)
#         plt.close() 
        
#     print(f"[Visualization] 成功生成 {dataset_name} 的 {T} 帧时序可视化结果，路径: {save_dir}")


# ################################################################
# # configs
# ################################################################
# parser = argparse.ArgumentParser(description='Training or pretraining for the same data type')

# parser.add_argument('--model', type=str, default='AFNO')
# parser.add_argument('--dataset',type=str, default='ns2d')

# parser.add_argument('--train_paths',nargs='+', type=str, default=[
#     'ns2d_fno_1e-5', 'ns2d_fno_1e-3', 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2', 'swe_pdb', 'dr_pdb', 'cfdbench'
# ])
# parser.add_argument('--test_paths',nargs='+',type=str, default=[
#     'ns2d_fno_1e-5', 'ns2d_fno_1e-3', 'ns2d_pdb_M1e-1_eta1e-2_zeta1e-2', 'swe_pdb', 'dr_pdb', 'cfdbench'
# ])
# parser.add_argument('--resume_path',type=str, default='/root/files/pdessl/logs_pretrain/AFNO_ns2d_1218_17_20_14:S_12_114400/model_99.pth')
# parser.add_argument('--ntrain_list', nargs='+', type=int, default=None)
# parser.add_argument('--ntest_list',nargs='+',type=int, default=None)
# parser.add_argument('--data_weights',nargs='+',type=int, default=[1])
# parser.add_argument('--use_writer', action='store_true',default=False)

# parser.add_argument('--res', type=int, default=128)
# parser.add_argument('--noise_scale',type=float, default=0.0)

# ### shared params
# parser.add_argument('--width', type=int, default=1024)
# parser.add_argument('--n_layers',type=int, default=6)
# parser.add_argument('--act',type=str, default='gelu')

# ### GNOT params
# parser.add_argument('--max_nodes',type=int, default=-1)

# ### FNO params
# parser.add_argument('--modes', type=int, default=20)
# parser.add_argument('--use_ln',type=int, default=0)
# parser.add_argument('--normalize',type=int, default=0)

# ### AFNO
# parser.add_argument('--patch_size',type=int, default=8)
# parser.add_argument('--n_blocks',type=int, default=8)
# parser.add_argument('--mlp_ratio',type=int, default=1)
# parser.add_argument('--out_layer_dim', type=int, default=32)

# parser.add_argument('--batch_size', type=int, default=10)
# parser.add_argument('--epochs', type=int, default=500)
# parser.add_argument('--lr', type=float, default=0.001)
# parser.add_argument('--opt',type=str, default='adam', choices=['adam','lamb'])
# parser.add_argument('--beta1',type=float,default=0.9)
# parser.add_argument('--beta2',type=float,default=0.999)
# parser.add_argument('--lr_method',type=str, default='step')
# parser.add_argument('--grad_clip',type=float, default=10000.0)
# parser.add_argument('--step_size', type=int, default=100)
# parser.add_argument('--step_gamma', type=float, default=0.5)
# parser.add_argument('--warmup_epochs',type=int, default=50)
# parser.add_argument('--sub', type=int, default=1)
# parser.add_argument('--T_in', type=int, default=10)
# parser.add_argument('--T_ar', type=int, default=1)
# parser.add_argument('--T_bundle', type=int, default=1)
# parser.add_argument('--gpu', type=str, default="3")
# parser.add_argument('--comment',type=str, default="")
# parser.add_argument('--log_path',type=str,default='')
# parser.add_argument('--n_channels',type=int, default=4)
# parser.add_argument('--n_class',type=int,default=12)

# args = parser.parse_args()
# device = torch.device("cuda:{}".format(args.gpu))
# print(f"Current working directory: {os.getcwd()}")


# ################################################################
# # load data and dataloader
# ################################################################
# train_paths = args.train_paths
# test_paths = args.test_paths
# args.data_weights = [1] * len(args.train_paths) if len(args.data_weights) == 1 else args.data_weights
# print('args',args)

# train_dataset = MixedTemporalDataset(args.train_paths, args.ntrain_list, res=args.res, t_in = args.T_in, t_ar = args.T_ar, normalize=False,train=True, data_weights=args.data_weights, n_channels=args.n_channels)
# test_datasets = [MixedTemporalDataset(test_path, res=args.res, n_channels = train_dataset.n_channels,t_in = args.T_in, t_ar=-1, normalize=False, train=False) for i, test_path in enumerate(test_paths)]
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
# test_loaders = [torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8) for test_dataset in test_datasets]
# ntrain, ntests = len(train_dataset), [len(test_dataset) for test_dataset in test_datasets]
# print('Train num {} test num {}'.format(train_dataset.n_sizes, ntests))

# ################################################################
# # load model
# ################################################################
# if args.model == "FNO":
#     model = FNO2d(args.modes, args.modes, args.width, img_size = args.res, patch_size=args.patch_size, in_timesteps = args.T_in, out_timesteps=1,normalize=args.normalize,n_layers = args.n_layers,use_ln = args.use_ln, n_channels=train_dataset.n_channels, n_cls=len(args.train_paths)).to(device)
# elif args.model == 'DPOT':
#     model = DPOTNet(img_size=args.res, patch_size=args.patch_size, in_channels=train_dataset.n_channels, in_timesteps = args.T_in, out_timesteps = args.T_bundle, out_channels=train_dataset.n_channels, normalize=args.normalize, embed_dim=args.width, modes=args.modes, depth=args.n_layers, n_blocks = args.n_blocks, mlp_ratio=args.mlp_ratio, out_layer_dim=args.out_layer_dim, act=args.act, n_cls=args.n_class).to(device)
# else:
#     raise NotImplementedError

# if args.resume_path:
#     print('Loading models and fine tune from {}'.format(args.resume_path))
#     load_model_from_checkpoint(model, torch.load(args.resume_path,map_location='cuda:{}'.format(args.gpu))['model'])

# #### set optimizer
# if args.opt == 'lamb':
#     optimizer = Lamb(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2), adam=True, debias=False,weight_decay=1e-4)
# else:
#     optimizer = Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=1e-6)

# # [Keeping your scheduler/log configuration exact...]
# # ... 
# comment = args.comment + '_{}_{}'.format(len(train_paths), ntrain)
# log_path = './logs/' + time.strftime('%m%d_%H_%M_%S') + comment if len(args.log_path)==0  else os.path.join('./logs',args.log_path + comment)
# if args.use_writer:
#     writer = SummaryWriter(log_dir=log_path)
#     fp = open(log_path + '/logs.txt', 'w+',buffering=1)
#     json.dump(vars(args), open(log_path + '/params.json', 'w'),indent=4)
#     sys.stdout = fp
# else:
#     writer = None
# print(model)
# count_parameters(model)

# ################################################################
# # Main function for pretraining & Evaluation
# ################################################################
# myloss = SimpleLpLoss(size_average=False)
# clsloss = torch.nn.CrossEntropyLoss(reduction='sum')

# test_l2_fulls, test_l2_steps = [], []
# time_test, total_samples = 0., 0

# model.eval()
# with torch.no_grad():
#     for id, test_loader in enumerate(test_loaders):
#         dataset_name = test_paths[id].split('/')[-1] # Extract just the name of the dataset
#         test_l2_full, test_l2_step = 0, 0
        
#         for batch_idx, (xx, yy, msk, _) in enumerate(test_loader):
#             loss = 0
#             xx = xx.to(device)
#             yy = yy.to(device)
#             msk = msk.to(device)

#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             time_i = time.time()

#             # Rollout prediction loop
#             for t in range(0, yy.shape[-2], args.T_bundle):
#                 y = yy[..., t:t + args.T_bundle, :]
#                 im, _ = model(xx)
#                 loss += myloss(im, y, mask=msk)

#                 if t == 0:
#                     pred = im
#                 else:
#                     pred = torch.cat((pred, im), -2)

#                 xx = torch.cat((xx[..., args.T_bundle:, :], im), dim=-2)

#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             time_test += time.time() - time_i

#             total_samples += xx.shape[0]
#             test_l2_step += loss.item()
#             test_l2_full += myloss(pred, yy, mask=msk)
            
#             # ----------------------------------------------------
#             # VISUALIZATION TRIGGER
#             # Visualize the prediction of the FIRST batch only
#             # ----------------------------------------------------
#             if batch_idx == 0:
#                 # 为每个 dataset 创建独立的子文件夹
#                 base_vis_dir = os.path.join(log_path, 'visualizations') if args.use_writer else './visualizations'
#                 dataset_vis_dir = os.path.join(base_vis_dir, dataset_name)
                
#                 visualize_all_timesteps(
#                     true_data=yy, 
#                     pred_data=pred, 
#                     dataset_name=dataset_name, 
#                     sample_idx=0,    # 仅可视化该 batch 的第一条序列
#                     save_dir=dataset_vis_dir
#                 )

#         test_l2_step_avg = test_l2_step / ntests[id] / (yy.shape[-2] / args.T_bundle)
#         test_l2_full_avg = test_l2_full / ntests[id]
#         test_l2_steps.append(test_l2_step_avg)
#         test_l2_fulls.append(test_l2_full_avg.item())

# print(test_l2_fulls)
# for i in range(len(test_paths)):
#     print('{}: {:.5f}'.format(test_paths[i], test_l2_fulls[i]))

# print('Total rollout time: {:.6f}s'.format(time_test))
# print('Total samples: {}'.format(int(total_samples)))
# print('Avg rollout time per sample: {:.6f}s'.format(time_test / total_samples))