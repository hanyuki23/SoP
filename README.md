# Non-collective Calibrating Strategy for Time Series Forecasting -[Paper Accepted by IJCAI 2025]

Deep learning-based approaches have demonstrated significant advancements in time series forecasting. Despite these ongoing developments, the complex dynamics of time series make it challenging to establish the rule of thumb for designing the golden model architecture. In this study, we argue that refining existing advanced models through a universal calibrating strategy can deliver substantial benefits with minimal resource costs, as opposed to elaborating and training a new model from scratch. We first identify a multi-target learning conflict in the calibrating process, which arises when optimizing variables across time steps, leading to the underutilization of the model’s learning capabilities. To address this issue, we propose an innovative calibrating strategy called Socket+Plug (SoP). This approach retains an exclusive optimizer and early-stopping monitor for each predicted target within each Plug while keeping the fully trained Socket backbone frozen. The model-agnostic nature of SoP allows it to directly calibrate the performance of any trained deep forecasting models, regardless of their specific architectures. Extensive experiments on various time series benchmarks and a spatio-temporal meteorological ERA5 dataset demonstrate the effectiveness of SoP, achieving up to a 22% improvement even when employing a simple MLP as the Plug.

<p align="center">
<img src=".\pic\models.png" height = "400" alt="" align=center />
</p>

<p align="center">
<img src=".\pic\sk.png" height = "400" alt="" align=center />
</p>

**Usage:**

Install Python 3.8. For convenience, execute the following command.


```
pip install -r requirements.txt

```

**Prepare Data:**

You can obtain the well pre-processed datasets from [https://github.com/thuml/Time-Series-Library]()

------------



**Train and evaluate model**

We provide the experiment scripts for Exchange dataset under the folder `./scrip/`. You can reproduce the experiment results as the following examples:

```
bash ./scrip/long_term_forecast/Exchange_script/iTransformer.sh

```
The specific operation steps are as follows：
- Training Socket model and set the following parameters in the iTransformer. sh file:

```
tunmodel=0 # Close step-wise SoP
cfintune=0 # Close variable-wise SoP

```
- Start training Plug:

```
tunmodel=1 # Open SoP as step wise SoP by default
cfintune=0 # Close variable-wise SoP

```
```
tunmodel=1 # Open SoP
cfintune=1 # Open variable-wise SoP

```
If you want to train multiple objectives (variables or time steps) in combination:

```
tunmodel=1 # Open SoP
cfintune=1 # Open variable-wise SoP
cseg_len=3 # Three variables are optimized together as a group, We refer to each such group of variables as an optimized Plug. 
```
Specifically, for a prediction target \(Y \in \mathbb{R}^{N \times S}\): If \(n\) variables along the \(N\)-dimension form an optimized plug to predict \(Y_{\text{plug}} \in \mathbb{R}^{n \times S}\), SoP creates the plug counts as \(M = \frac{N}{n}\).

------------


**Develop your own model.**

Add the model file to the folder ./models. You can follow the ./models/Transformer.py.
Include the newly added model in the Exp_Basic.model_dict of ./exp/exp_basic.py.
Create the corresponding scripts under the folder ./scrip.

**Citation**

If you find this repo useful, please cite our paper.


```

@article{wang2025non,
  title={Non-collective Calibrating Strategy for Time Series Forecasting},
  author={Wang, Bin and Han, Yongqi and Ma, Minbo and Li, Tianrui and Zhang, Junbo and Hong, Feng and Yu, Yanwei},
  journal={arXiv preprint arXiv:2506.03176},
  year={2025}
}

```

**Contact**

Yongqi Han (hanyuki23@stu.ouc.edu.cn)
