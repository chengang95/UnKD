# UnKD
This is an implemention for our WSDM 2023 paper "Unbiased Knowledge Distillation for Recommendation" based on Pytorch.

## Requirements
+ pytorch == 1.8.0
+ Numpy
+ python3
+ pandas == 0.25.0
+ tqdm

## Datasets
Data preprocessing through python preprocess_xxx.py
+ Movielens:[original data](https://grouplens.org/datasets/movielens/);  Data preprocessing code：preprocess_ml1m.py
+ CiteULike:[original data](https://github.com/changun/CollMetric); Data preprocessing code：preprocess_cite.py
+ Apps:[original data](http://jmcauley.ucsd.edu/data/amazon/links.html); Data preprocessing code：preprocess_app.py

## Parameters
Key parameters in train_new_api.py:
+ --split_group: The number of divided groups.
+ --sample_num: The number of samples.
+ --recdim: Model dimensions.
+ -- model: MF or LightGCN.
+ --teacher_model: MF or LightGCN.
+ --teacher_dim: Teacher model dimensions.
+ --sampler: Distillation method, such as: UnKD.
+ --datasetpkl: Dataset save location.
+ others: read help, or "python xxx.py --help"

## Commands 
We provide following commands for our models and baselines.
We provide two methods.
#### 1. Simply Reproduce the Results:
+ First, we have provided the model that we trained for simply reproducing the results in our paper. Then you can run the following commands to reproduce the results in our paper.
  ```
  python -u  distill_new_api.py   --split_group=4 --sample_num=30   --dataset=Apps   --datasetpkl=$location   --recdim=10  --model=MF  --teacher_model=MF  --decay=0.01   --epochs=1000 --lr=0.01  --seed=2022   --stopflag=ndcg   --teacher_dim=100  --kd_weight=1.0   --sampler=UnKD
  ```
 
## Citation
If you use our codes in your research, please cite our paper.
