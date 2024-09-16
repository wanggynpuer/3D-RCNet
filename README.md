### [3D-RCNet:Learning from Transformer to Build a 3D Relational ConvNet for Hyperspectral Image Classification](https://arxiv.org/abs/2408.13728)

<p align="center">
<a href="https://arxiv.org/search/cs?searchtype=author&query=Jing,+H">Haizhao Jing</a>, 
<a href="https://arxiv.org/search/cs?searchtype=author&query=Wan,+L">Liuwei Wan</a>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Xue,+X"> Xizhe Xue</a>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Zhang,+H"> Haokui Zhang</a>,
<a href="https://arxiv.org/search/cs?searchtype=author&query=Li,+Y">Ying Li</a>,
</p><br>

<br>

### The 3D-RCNet framework

<img src="./assets/Fig1.png" alt="description" width="100%">

**Fig1. The 3D-RCNet framework proposed by us, and the framework uses four stages of blocks for feature extraction at different depths on HSI data**<br>

<br>

### Comparison of the three methods

<img src="./assets/Fig2.png" alt="description" width="85%">

**Fig2. Comparison of the three methods, the total MACs required by each method with the same input. (a) is 3D-ConvBlock,(b) is Self-attention, and (c) is our proposed 3D-RCBlock. **<br>

<br>

<img src="./assets/table1.png" alt="description" width="100%">

<br>

### Directory and File Structure

```
./                                            # current (project) directory
│
├── assets									  # figures and tables 
│
├── data/                                     # Files to be processed in the dataset
│   └── HSI_datasets/
│       ├── data_h5/
│       └── samples/
├── data_preprocess/
│   ├── data_list/                            # The preprocessed data is placed in the data_list folder.
		├──Indian_pines_split.txt
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

**🔥🔥🔥Note:** The `Indian_pines.txt`, `Indian_pines_test.txt`, and `Indian_pines_train.txt` files generated in the `data_list` directory are created by executing `mat_2_h5.py` and `preprocessing.py` in sequence.🔥🔥🔥<br>

**The `data` folder contains the datasets to be processed**

**`data_preprocess` folder:**

The `data_list` folder contains preprocessed data.

- `mat_2_h5.py`: Dataset format conversion
- `preprocess.py`: Data preprocessing
  - `functions_for_samples_extraction.py`

**`training`folder:**

The `models` folder contains our proposed 3D-RCNet.

- `get_cls_map.py`: Generate pseudo-color composite images
- `main_cv_paper.py`: Training script
  - functions_for_training.py
  - functions_for_evaluating.py



<br>

## **Environment Setup and Installation**

`python: 3.11`

**NOTE: Latest PyTorch requires Python 3.8 or later.**