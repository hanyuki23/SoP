# Non-collective Calibrating Strategy for Time Series Forecasting 

Deep learning-based approaches have demonstrated significant advancements in time series forecasting. Despite these ongoing developments, the complex dynamics of time series make it challenging to establish the rule of thumb for designing the golden model architecture. In this study, we argue that refining existing advanced models through a universal calibrating strategy can deliver substantial benefits with minimal resource costs, as opposed to elaborating and training a new model from scratch. We first identify a multi-target learning conflict in the calibrating process, which arises when optimizing variables across time steps, leading to the underutilization of the model’s learning capabilities. To address this issue, we propose an innovative calibrating strategy called Socket+Plug (SoP). This approach retains an exclusive optimizer and early-stopping monitor for each predicted target within each Plug while keeping the fully trained Socket backbone frozen. The model-agnostic nature of SoP allows it to directly calibrate the performance of any trained deep forecasting models, regardless of their specific architectures. Extensive experiments on various time series benchmarks and a spatio-temporal meteorological ERA5 dataset demonstrate the effectiveness of SoP, achieving up to a 22% improvement even when employing a simple MLP as the Plug.

<p align="center">
<img src=".\pic\models.png" height = "400" alt="" align=center />
</p>

**Usage:**

Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt

```

**Prepare Data:**

You can obtain the well pre-processed datasets from [https://github.com/thuml/Time-Series-Library]()

**Train and evaluate model**

We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh

```
The main arguments in config.json are described below:

tun_model:use step-wise SoP

channel_fintune: use variable-wise SoP

cseg_len: when you want to calibrate three variables or three time steps together, set the cseg_1en parameter to 3

eg. If you want to run Exchange_96_96 and use a variable-wise SoP, the parameter settings are: tun_madel=1, channel_fintune=1, cseg_len=1, means that the predicted results of each variable will have a corresponding plug for calitration. 

If you want to run Exchange_96_96 and use a step-wise SoP, the parameter settings are: tun_madel=1, channel_fintune=0, cseg_len=1, means that the predicted results of each step will have a corresponding plug for calitration. 


**Develop your own model.**

Add the model file to the folder ./models. You can follow the ./models/Transformer.py.
Include the newly added model in the Exp_Basic.model_dict of ./exp/exp_basic.py.
Create the corresponding scripts under the folder ./scripts.

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


**Acknowledgement**

This project is supported by the National Natural Science Foundation of China (Nos. 62402463, 41976185, 62176243, 62176221 and 72242106.

**Contact**

Yongqi Han (hanyuki23@stu.ouc.edu.cn)
