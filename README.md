## DPOT：用于大规模偏微分方程(PDE)预训练的自回归去噪算子Transformer (ICML'2024)

这是[论文](https://arxiv.org/pdf/2403.03542) DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training (ICML'2024) 的官方代码。它在多个偏微分方程(PDE)数据集上预训练了神经算子Transformer（参数量从 **7M** 到 **1B**）。预训练权重可以在 https://huggingface.co/hzk17/DPOT 找到。

![fig1](/resources/dpot.jpg)

我们预训练的 DPOT 在多个 PDE 数据集上实现了最先进（state-of-the-art）的性能，并可用于在不同类型的下游 PDE 问题上进行微调。

![fig2](/resources/dpot_result.jpg)



### 使用方法 

##### 预训练模型

我们提供了五种不同大小的预训练检查点（checkpoints）。预训练权重位于 https://huggingface.co/hzk17/DPOT。

| 规模   | 注意力维度 | MLP维度 | 层数 | 头数 | 模型大小 |
| ------ | ------------- | ------- | ------ | ----- | ---------- |
| Tiny   | 512           | 512     | 4      | 4     | 7M         |
| Small  | 1024          | 1024    | 6      | 8     | 30M        |
| Medium | 1024          | 4096    | 12     | 8     | 122M       |
| Large  | 1536          | 6144    | 24     | 16    | 509M       |
| Huge   | 2048          | 8092    | 27     | 8     | 1.03B      |

以下是加载预训练模型的示例代码：
```python
model = DPOTNet(img_size=128, patch_size=8, mixing_type='afno', in_channels=4, in_timesteps=10, out_timesteps=1, out_channels=4, normalize=False, embed_dim=512, modes=32, depth=4, n_blocks=4, mlp_ratio=1, out_layer_dim=32, n_cls=12)
model.load_state_dict(torch.load('model_Ti.pth')['model'])
```



##### 数据集协议

所有数据集均使用 hdf5 格式存储，包含 `data` 字段。部分数据集以单独的 hdf5 文件存储，其他数据集则存储在单个 hdf5 文件中。

在 `data_generation/preprocess.py` 中，我们提供了处理各个来源数据集的脚本。请从这些来源下载原始文件，并将它们预处理到 `/data` 文件夹中。

| 数据集       | 链接                                                         |
| ------------- | ------------------------------------------------------------ |
| FNO 数据      | [点击这里](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| PDEBench 数据 | [点击这里](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) |
| PDEArena 数据 | [点击这里](https://microsoft.github.io/pdearena/datadownload/)   |
| CFDbench 数据 | [点击这里](https://cloud.tsinghua.edu.cn/d/435413b55dea434297d1/) |

在 `utils/make_master_file.py` 中，我们包含了所有的数据集配置。当合并新的数据集时，你应该添加一个配置字典（configuration dict）。它存储了所有的相对路径，因此你可以在任何地方运行。 

```bash
mkdir data
```

##### 单 GPU 预训练

现在我们有一个单 GPU 预训练代码脚本 `train_temporal.py`，你可以通过以下命令启动它

```bash
python train_temporal.py --model DPOT --train_paths ns2d_fno_1e-5 --test_paths ns2d_fno_1e-5 --gpu 0 
```

来开启一个训练过程。

或者，你可以在 `configs/ns2d.yaml` 中编写一个配置文件，并通过以下命令自动使用空闲 GPU 启动它：

```bash
python trainer.py --config_file ns2d.yaml
```

##### 多 GPU 预训练

```bash
python parallel_trainer.py --config_file ns2d_parallel.yaml
```

##### 配置文件

目前我使用 yaml 作为配置文件。你可以为 args 指定参数。如果你想运行多个任务，你可以将参数移动到 `tasks` 节点中：

```yaml
model: DPOT
width: 512
tasks:
 lr: [0.001,0.0001]
 batch_size: [256, 32] 
```

这意味着如果你将此配置提交给 `trainer.py`，你将会启动 2 个任务。 

##### 环境要求


```bash
python -m pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install matplotlib scikit-learn scipy pandas h5py
python -m pip install timm einops tensorboard
```

### 代码结构

- `README.md`
- `train_temporal.py`: 单 GPU 预训练自回归模型的主代码
- `trainer.py`: 用于自动调度训练任务以进行参数调优的框架
- `utils/`
  - `criterion.py`: 相对误差的损失函数
  - `griddataset.py`: 混合时间均匀网格数据集的数据集类
  - `make_master_file.py`: 数据集配置文件
  - `normalizer`: 归一化方法（#TODO: 实现实例可逆归一化）
  - `optimizer`: 支持复数的 Adam/AdamW/Lamb 优化器
  - `utilities.py`: 其他辅助函数
- `configs/`: 预训练或微调的配置文件
- `models/`
  - `dpot.py`:         DPOT 模型
  - `fno.py`:          带组归一化（group normalization）的 FNO
  - `mlp.py`
- `data_generation/`:  一些预处理数据的代码（如果你想使用它们请询问 hzk）
  - `darcy/`
  - `ns2d/`



### 引用

如果你在研究中使用了 DPOT，请使用以下 BibTeX 条目。

```bibtex
@article{hao2024dpot,
  title={DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training},
  author={Hao, Zhongkai and Su, Chang and Liu, Songming and Berner, Julius and Ying, Chengyang and Su, Hang and Anandkumar, Anima and Song, Jian and Zhu, Jun},
  journal={arXiv preprint arXiv:2403.03542},
  year={2024}
}
```