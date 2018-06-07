# Sequence Generation Model with Global Embedding
- This is the code for our paper *SGM: Sequence Generation Model for Multi-label Classification*
- Be careful: the provided codes are based on the RCV1-V2 dataset. If you need to run the codes on other datasets, please correspondly modify all program statements that relate to the specific name of the dataset.

***********************************************************

## Datasets
* RCV1-V2
* AAPD

Two datasets are available at https://drive.google.com/file/d/1DbYWVXt_J_6wAgjzVh9LG_MG1UCjBQuS/view?usp=sharing

***************************************************************

## Requirements
* Ubuntu 16.0.4
* Python 3.5
* Pytorch 0.3.1

***************************************************************

## Reproducibility
We provide the pretrained checkpoints of the SGM model and the SGM+GE model on the RCV1-V2 dataset to help you reproduce our reproted experimental results. The detailed reproduction steps are as follows:

- Please download the RCV1-V2 dataset and checkpoints first, then put them in the folder *./data/data/*
- Preprocessing: 
```
python3 preprocess.py 
```
- Predict:
```
python3 predict.py -gpus id -log log_name
```


***************************************************************

## Preprocessing
```
python3 preprocess.py 
```
Remember to put the data into a folder and name them *train.src*, *train.tgt*, *valid.src*, *valid.tgt*, *test.src* and *test.tgt*, and make a new folder inside called *data*

***************************************************************

## Training
```
python3 train.py -gpus id -log log_name
```

****************************************************************

## Evaluation
```
python3 predict.py -gpus id -restore checkpoint -log log_name
```

*******************************************************************

## Citation
If you use the above codes or the AAPD dataset we built for your research, please cite the paper:

```
@inproceedings{YangCOLING2018,
   author = {Pengcheng Yang and Xu Sun and Wei Li and Shuming Ma and Wei Wu and Houfeng Wang},
   title = {SGM: Sequence Generation Model for Multi-label Classification},
   booktitle = {{COLING} 2018},
   year = {2018}
}
```
