# ASTNN--A Novel Neural Source Code Representation based on Abstract Syntax Tree
This repository includes the code and experimental data in our paper that will be presented at ICSE'2019. It can be used to encode code fragments into supervised vectors for various tasks such as source code classification and code clone detection. 

### Requirements
+ python 3.6<br>
+ pandas 0.20.3<br>
+ gensim 3.5.0<br>
+ scikit-learn 0.19.1<br>
+ pytorch 0.3.1<br>
+ pycparser 2.18<br>
+ javalang 0.11.0<br>
+ GPU with CUDA support is also needed

### How to install
Install all the dependent packages via pip:

	$ pip install pandas==0.20.3 gensim==3.5.0 scikit-learn==0.19.1 pycparser==2.18 javalang==0.11.0
 
Install pytorch 0.3.1: 

	$ pip install https://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl

The version of pytorch 0.3.1 is mandatory. Higher versions may lead to errors for our exisiting code, and we will improve our code in the future.

### Source Code Classification
1. `cd astnn`
2. run `python pipeline.py` to generate preprocessed data.
3. run `python train.py` for training and evaluation

### Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.
