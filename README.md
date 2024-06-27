# 3D-RCNet

**目录和文件结构**：

```
./                                            # current (project) directory
├── data/                                     # Files to be processed in the dataset
│   └── HSI_datasets/
│       ├── data_h5/
│       └── samples/
├── data_preprocess/
│   ├── data_list/                            # The preprocessed data is placed in the data_list folder.
│   ├── functions_for_samples_extraction.py
│   ├── mat_2_h5.py                           # Dataset format conversion
│   └── preprocess.py                         # Preprocessing the dataset
└── training/
    ├── models/
    ├── functions_for_evaluating.py
    ├── functions_for_training.py
    ├── get_cls_map.py                        # Generating pseudocolored synthesized images
    └── main_cv_paper.py
```

**data文件夹中放置的是待处理的数据集文件**

**data_preprocess文件夹中包含：**

data_list文件夹中放置preprocess后的数据

- mat_2_h5.py：数据集格式转换

- preprocess.py：预处理数据集

  - functions_for_samples_extraction.py

**training文件夹中包含：**

models文件夹中放置我们提出的3D-RCNet

- get_cls_map.py: 获取伪彩合成图
- main_cv_paper.py: 训练程序
  - functions_for_training.py
  - functions_for_evaluating.py

## 环境配置与安装

`python版本: 3.11`

[python下载安装](https://www.python.org/downloads/)


[pytorch下载安装](https://pytorch.org/)

**NOTE: Latest PyTorch requires Python 3.8 or later.**

### 使用 pip 安装pytorch

#### 安装 CPU 版本

`pip3 install torch torchvision torchaudio`

#### 安装 GPU 版本

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
