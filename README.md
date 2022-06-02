# Paddle CVPR 2022 NAS competition track 2: Performance Estimation
Repo for Paddle CVPR 2022 NAS track 2 performance estimation, [link](https://aistudio.baidu.com/aistudio/competition/detail/150/0/introduction).

## Implementation
1. One-hot encoding for each architecture, $E(\alpha)\in R^{37\times10}$, we pad with zeros for arch with depth less then 12.
2. Positional encoding is added to each arch encoding for preserving the depth information.
3. Predictor is based on MLP, the final output layer outputs 1 score for each arch.
4. We adopt ranking-based loss to regulate the relative ranking for predicted scores, see [ReNAS](https://arxiv.org/abs/1910.01523).
5. We traverse the predictors according to their top ktau over test split and upload to the eval server.
## Prerequisite

For test predictors with trained weights, please first download the checkpoints from Baidu [link](https://pan.baidu.com/s/1PKNiuHyMcxrIBCCK_Hgx3g), password is `n06d`; then place the checkpoints in the `results/` folder and train and test data in the `data` folder.

In summary, the folder hierarchy follow:
```
.
├── dataset.py
├── main.py
├── network.py
├── README.md
├── test.ipynb
├── test.sh
├── train.ipynb
├── train.sh
├── results
│   ├── cplfw_final.pth
│   ├── dukemtmc_final.pth
│   ├── market1501_final.pth
│   ├── msmt17_final.pth
│   ├── sop_final.pth
│   ├── vehicleid_final.pth
│   ├── veri_final.pth
│   └── veriwild_final.pth
├── data
│   ├── CVPR_2022_NAS_Track2_test.json
│   └── CVPR_2022_NAS_Track2_train.json
└── test.py
```
## Reproduce
1. For training the predictors, directly run the `train.sh` or run the cell in `train.ipynb`
2. For testing and obtaining the final ranking over 99500 archs, make sure you download the checkpoints and store in the right folder mentioned above, then run the `test.sh` or `test.ipynb`.
3. Notice, the results may exist small discrepancies due to codes lino and running environments.  