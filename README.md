# Non-collective Calibrating Strategy for Time Series Forecasting -[Paper Accepted by IJCAI 2025]

###### 🚀 Enhance pretrained time series models without full retraining!

SoP is a universal calibration strategy that resolves multi-target learning conflicts by optimizing each prediction target independently while keeping the backbone frozen. Achieves **up to 22%** improvement even with simple MLP Plugs.

<p align="center">
<img src=".\pic\models.png" height = "400" alt="" align=center />
</p>

<p align="center">
<img src=".\pic\sk.png" height = "400" alt="" align=center />
</p>


------------



**Usage:**

Install Python 3.8. For convenience, execute the following command.


```
pip install -r requirements.txt

```

📂 **Prepare Data:**

You can obtain the well pre-processed datasets from [https://github.com/thuml/Time-Series-Library]()

------------

##### 🚀 Quick Start  

**Train and evaluate model**

We provide the experiment scripts for Exchange dataset under the folder `./scrip/`. You can reproduce the experiment results as the following examples:

```
bash ./scrip/long_term_forecast/Exchange_script/iTransformer.sh

```
The specific operation steps are as follows：
- Training Socket model and set the following parameters in the iTransformer. sh file:

```
tunmodel=0 # Not using SoP for calibration
cfintune=0

```
- Start training Plug:

```
tunmodel=1 # Using SoP for calibration
cfintune=0 # Using step-wise SoP
```

```
tunmodel=1 # Using SoP for calibration
cfintune=1 # Using variable-wise SoP
```

If you want to train multiple objectives (variables or time steps) in combination:

```
tunmodel=1 # Using SoP for calibration
cfintune=1 # Using variable-wise SoP
cseg_len=3 # Three variables are optimized together as a group, We refer to each such group of variables as an optimized Plug
```

##### Training Modes

| Mode | `tunmodel` | `cfintune` | Description |
|------|------------|------------|-------------|
| **Baseline (No SoP)** | `0` | `0` | Train Socket model only |
| **Step-wise SoP** | `1` | `0` | Optimize per time step |
| **Variable-wise SoP** | `1` | `1` | Optimize per variable |
| **Grouped Steps** | `1` | `0` + `cseg_len=n` | Optimize `n` stps jointly |
| **Grouped Variables** | `1` | `1` + `cseg_len=n` | Optimize `n` variables jointly |

Specifically, for a prediction target $Y \in \mathbb{R}^{N \times S}$: If $n$ variables along the $N$ dimension form an optimized plug to predict $Y_{\text{plug}}$ $\in \mathbb{R}^{n \times S}$ , SoP creates the plug counts as $M = \frac{N}{n}$.

------------


🛠️ **Develop your own model.**

Add the model file to the folder ./models. You can follow the ./models/Transformer.py.
Include the newly added model in the Exp_Basic.model_dict of ./exp/exp_basic.py.
Create the corresponding scripts under the folder ./scrip.


------------



📜 **Citation**

If you find this repo useful, please cite our paper.


```
@article{wang2025non,
  title={Non-collective Calibrating Strategy for Time Series Forecasting},
  author={Wang, Bin and Han, Yongqi and Ma, Minbo and Li, Tianrui and Zhang, Junbo and Hong, Feng and Yu, Yanwei},
  journal={arXiv preprint arXiv:2506.03176},
  year={2025}
}

```

📩 **Contact**

Yongqi Han (hanyuki23@stu.ouc.edu.cn)
