### Requirements
python 3.6<br>
pandas 0.20.3<br>
gensim 3.5.0<br>
scikit-learn 0.19.1<br>
pytorch 0.3.1<br>
pycparser 2.18<br>
javalang 0.11.0<br>
GPU with CUDA support is also needed

### Source Code Classification
1. `cd astnn`
2. run `python pipeline.py` to generate preprocessed data.
3. run `python train.py` for training and evaluation

### Code Clone Detection

 1. `cd clone`
 2. run `python pipeline.py --lang c` or `python pipeline.py --lang java` to generate preprocessed data for the two datasets.
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.
