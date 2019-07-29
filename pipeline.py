import pandas as pd
import os

class Pipeline:
    def __init__(self,  ratio, root):
        #ratio = '3:1:1' root = 'data/'
        self.ratio = ratio
        self.root = root
        self.sources = None
        self.train_file_path = None
        self.dev_file_path = None
        self.test_file_path = None
        self.size = None

    # parse source code
    def parse_source(self, output_file, option):
        #output_file = 'ast.pkl' option = 'existing'
        path = self.root+output_file
        #path = 'data/ast.pkl'
        if os.path.exists(path) and option is 'existing':
            source = pd.read_pickle(path)
        else:
            from pycparser import c_parser
            parser = c_parser.CParser()
            source = pd.read_pickle(self.root+'programs.pkl')
            #source file is 'data/programs.pkl'

            #***********************************************************
# 0          0  int main()\n{\n\tint a;\n\tint bai,wushi,ershi...     97
# 1          1  int main()\n{\n    int m,x100,x50,x20,x10,x5,x...     97
# 2          2  int main()\n{\n    int n,i,shuzu[111],count1=0...     97
# 3          3  int main()\n{\n\tint n,a1=0,a2=0,a3=0,a4=0,a5=...     97
# 4          4  int main()\n{\n\tint n,a,b,c,d,e,f;\n\ta=0;b=0...     97
# 5          5  \n\nint main()\n{\n\tint input,hundred,fifty,t...     97
# 6          6  \n\n\n\n\n\n\n\n\nint main()                  ...     97
# 7          7  int main()\n{\n\tint n,i,j=1,a[6];\n\t\n\tscan...     97
# 8          8    int main(){\n   int n,x;\n   scanf("%d",&n);...     97
# 9          9  \nint main()\n{\n\tint i,n;\n\tint a[10];\n\ti...     97
# 10        10  int main()\n{\n   int n,a,b,c,d,e,f,l,m;\n   s...     97
            #***********************************************************

            source.columns = ['id', 'code', 'label']
            source['code'] = source['code'].apply(parser.parse)
            #read data from 'data/programs.pkl' and use cparser to transform the code to AST file, then store it into 'data/ast.pkl'

            #***********************************************************
# 0          0  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 1          1  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 2          2  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 3          3  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 4          4  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 5          5  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 6          6  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 7          7  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 8          8  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 9          9  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
# 10        10  FileAST(ext=[FuncDef(decl=Decl(name='main',\n ...     97
            #***********************************************************

            source.to_pickle(path)
        self.sources = source
        return source

    # split data for training, developing and testing
    def split_data(self):
        data = self.sources
        #data_num = 52001
        data_num = len(data)
        #self.ratio = '3:1:1'
        #ratios = [3, 1, 1]  It's for decide the percentage of the training data and validation data.
        ratios = [int(r) for r in self.ratio.split(':')]
        #training dataset has 3/5 data.
        train_split = int(ratios[0]/sum(ratios)*data_num)
        #validation dataset has 1/5 data. (the number is to 4/5)
        val_split = train_split + int(ratios[1]/sum(ratios)*data_num)
        #frac represent the pencentage of the data choose, 1 is 100%
        #random_state represent the data choose method, 1 represent the data could be duplicated and 0 represent unduplicated.
        data = data.sample(frac=1, random_state=666)
        #iloc : pick up the data by rows.
        train = data.iloc[:train_split] #3/5
        dev = data.iloc[train_split:val_split] #1/5
        test = data.iloc[val_split:] #1/5

        def check_or_create(path):
            if not os.path.exists(path):
                os.mkdir(path)
        #train_path = 'data/train/'
        train_path = self.root+'train/'
        check_or_create(train_path)
        #train_file_path = 'data/train/train_.pkl'
        self.train_file_path = train_path+'train_.pkl'
        train.to_pickle(self.train_file_path)

        #dev_path = 'data/dev/'
        dev_path = self.root+'dev/'
        check_or_create(dev_path)
        #dev_file_path = 'data/dev/dev_.pkl'
        self.dev_file_path = dev_path+'dev_.pkl'
        dev.to_pickle(self.dev_file_path)

        #test_path = 'data/test/'
        test_path = self.root+'test/'
        check_or_create(test_path)
        #test_file_path = 'data/test/test_.pkl'
        self.test_file_path = test_path+'test_.pkl'
        test.to_pickle(self.test_file_path)

    # construct dictionary and train word embedding
    def dictionary_and_embedding(self, input_file, size):
        #input_file = None size = 128
        self.size = size
        if not input_file:
            #input_file = 'data/train/train_.pkl'
            input_file = self.train_file_path
        trees = pd.read_pickle(input_file)
        if not os.path.exists(self.root+'train/embedding'):
            os.mkdir(self.root+'train/embedding')
        #prepare_data.py
        from prepare_data import get_sequences

        def trans_to_sequences(ast):
            sequence = []
            get_sequences(ast, sequence)
            return sequence
        corpus = trees['code'].apply(trans_to_sequences)
        str_corpus = [' '.join(c) for c in corpus]
        trees['code'] = pd.Series(str_corpus)
        trees.to_csv(self.root+'train/programs_ns.tsv')

        from gensim.models.word2vec import Word2Vec
        #size is the dimension of the embedding. 
        #workers control the training rows. 
        #sg = 1 represent Skip-Gram model. (Skip-Gram : input word -> context. CBOW : context -> input word)
        #min_count represent the minimum count of the words.
        w2v = Word2Vec(corpus, size=size, workers=16, sg=1, min_count=3)
        w2v.save(self.root+'train/embedding/node_w2v_' + str(size))

    # generate block sequences with index representations
    def generate_block_seqs(self,data_path,part):
        from prepare_data import get_blocks as func
        from gensim.models.word2vec import Word2Vec

        word2vec = Word2Vec.load(self.root+'train/embedding/node_w2v_' + str(self.size)).wv
        vocab = word2vec.vocab
        max_token = word2vec.syn0.shape[0]

        def tree_to_index(node):
            token = node.token
            result = [vocab[token].index if token in vocab else max_token]
            children = node.children
            for child in children:
                result.append(tree_to_index(child))
            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree
        trees = pd.read_pickle(data_path)
        trees['code'] = trees['code'].apply(trans2seq)
        trees.to_pickle(self.root+part+'/blocks.pkl')

    # run for processing data to train
    def run(self):
        print('parse source code...')
        self.parse_source(output_file='ast.pkl',option='existing')
        print('split data...')
        self.split_data()
        print('train word embedding...')
        self.dictionary_and_embedding(None,128)
        print('generate block sequences...')
        self.generate_block_seqs(self.train_file_path, 'train')
        self.generate_block_seqs(self.dev_file_path, 'dev')
        self.generate_block_seqs(self.test_file_path, 'test')


ppl = Pipeline('3:1:1', 'data/')
ppl.run()


