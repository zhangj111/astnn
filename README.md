# ASTNN--A Novel Neural Source Code Representation based on Abstract Syntax Tree
This repository includes the code and experimental data in our paper entitled "A Novel Neural Source Code Representation based on Abstract Syntax Tree" that will be presented at ICSE'2019. It can be used to encode code fragments into supervised vectors for various source code related tasks. We have applied our neural source code representation to two common tasks: source code classification and code clone detection. It is also expected to be helpful in more tasks.

### Requirements
+ python 3.6<br>
+ pandas 0.20.3<br>
+ gensim 3.5.0<br>
+ scikit-learn 0.19.1<br>
+ pytorch 0.3.1<br>
+ pycparser 2.18<br>
+ javalang 0.11.0<br>
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE should be configured based on the GPU memory size

### How to install
Install all the dependent packages via conda and pip:

	$ conda install pandas scikit-learn pycparser paramiko -c anaconda gensim
	$ pip install --no-deps javalang
 
Install pytorch: 

	$ conda install -c pytorch pytorch

The version of pytorch >= 0.3.1 is mandatory. Higher versions may lead to errors for our exisiting code, and we will improve our code in the future. See https://pytorch.org/get-started/previous-versions/ for more options according to your CUDA version.

### Source Code Classification
1. `cd astnn`
2. run `python pipeline.py` to generate preprocessed data.
3. run `python train.py` for training and evaluation

### Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.
