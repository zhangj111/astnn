### Requirements
python 3.6
pandas 0.20.3
gensim 3.5.0
scikit-learn 0.19.1
pytorch 0.3.1
pycparser 2.18
javalang 0.11.0
GPU with CUDA support is also needed

### Source Code Classification
1. `cd code_emb`
2. run `python train.py`
3. (optional) You can run `python pipeline.py` to regenerate preprocessed data and then redo step 2.

### Code Clone Detection

 1. `cd clone`
 2. run `python train.py --lang c` to train on OJClone, `python train.py --lang java` on BigCLoneBench respectively.
 3. (optional) run `pipeline.py --lang c` or `pipeline.py --lang java` to regenerate preprocessed data and then redo step 2.