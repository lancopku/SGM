# Sequence Generation Model with Global Embedding
This is the code for our paper *SGM: Sequence Generation Model for Multi-label Classification*

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
