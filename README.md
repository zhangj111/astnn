# ASTNN--A Novel Neural Source Code Representation based on Abstract Syntax Tree
This repository includes the code and experimental data in our paper entitled "A Novel Neural Source Code Representation based on Abstract Syntax Tree" published in ICSE'2019. It can be used to encode code fragments into supervised vectors for various source code related tasks. We have applied our neural source code representation to two common tasks: source code classification and code clone detection. It is also expected to be helpful in more tasks.

### Requirements
+ python 3.6<br>
+ pandas 0.20.3<br>
+ gensim 3.5.0<br>
+ scikit-learn 0.19.1<br>
+ pytorch 1.0.0<br> (The version used in our paper is 0.3.1 and source code can be cloned by specifying the v1.0.0 tag if needed)
+ pycparser 2.18<br>
+ javalang 0.11.0<br>
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE should be configured based on the GPU memory size

### How to install
Install all the dependent packages via pip:

	$ pip install pandas==0.20.3 gensim==3.5.0 scikit-learn==0.19.1 pycparser==2.18 javalang==0.11.0
 
Install pytorch according to your environment, see https://pytorch.org/ 


### Source Code Classification
1. `cd astnn`
2. run `python pipeline.py` to generate preprocessed data.
3. run `python train.py` for training and evaluation

### Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.
 
### Citation
  If you find this code useful in your research, please, consider citing our paper:
  > @inproceedings{Zhang:2019:NNS:3339505.3339604,
 author = {Zhang, Jian and Wang, Xu and Zhang, Hongyu and Sun, Hailong and Wang, Kaixuan and Liu, Xudong},
 title = {A Novel Neural Source Code Representation Based on Abstract Syntax Tree},
 booktitle = {Proceedings of the 41st International Conference on Software Engineering},
 series = {ICSE '19},
 year = {2019},
 location = {Montreal, Quebec, Canada},
 pages = {783--794},
 numpages = {12},
 url = {https://doi.org/10.1109/ICSE.2019.00086},
 doi = {10.1109/ICSE.2019.00086},
 acmid = {3339604},
 publisher = {IEEE Press},
 address = {Piscataway, NJ, USA},
 keywords = {abstract syntax tree, code classification, code clone detection, neural network, source code representation},
}
