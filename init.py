import Classifiers as clf
import extract as extr
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
import pdb
from random import shuffle
from collections import defaultdict 
from unidecode import unidecode

from gensim.parsing.preprocessing import STOPWORDS as stopwords_gensim
from nltk.corpus import stopwords as s_words           
from nltk.tokenize.moses import MosesTokenizer         
from keras.preprocessing.text import text_to_word_sequence as keras_tokenize
import nltk
import string
import gensim

import dl_models
import keras

STOPWORDS = set(s_words.words('english')).union(set(stopwords_gensim))
PUNCTUATIONS = set([w for w in string.punctuation])
GLOVE_MODEL_FILE = "/home/pinkesh.badjatiya/WORD_EMBEDDINGS/GENSIM.glove.840B.300d.txt"
TOKENIZER = MosesTokenizer()
EMBEDDING_DIM = 300

#word2vec_model = None
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(GLOVE_MODEL_FILE)                                                                


def pad_sentences(sentences, word2id_map):
    SEN_LEN = 10
    out = []
    for sen in sentences:
        if len(sen) < SEN_LEN:
            out.append(sen + [word2id_map['UNK']]*(SEN_LEN - len(sen)))
        elif len(sen) > SEN_LEN:
            out.append(sen[:SEN_LEN])
        else:
            out.append(sen)
    return out
        


def tokenize(text):
    return TOKENIZER.tokenize(text)

def get_embedding_weights(initialize, word2id_map):
    assert initialize in ('random', 'word2vec')
    embedding = np.zeros((len(word2id_map) + 1, EMBEDDING_DIM))
    n = 0                        
    for k, v in word2id_map.iteritems():
        try:                     
            if initialize == 'word2vec':
                embedding[v] = word2vec_model[k]                                                                                                          
            elif initialize == 'random':
                embedding[v] = np.random.rand(EMBEDDING_DIM)
            else:                
                print 'ERROR: Do not know how to initalize the embedding matrix. Initializing with ZEROS'
                embedding[v] = np.zeros((EMBEDDING_DIM,))
                                 
        except:                  
            n += 1               
            pass                 
    print "%d embeddings missed"%n
    return embedding   

def _gen_vocab(tokenizer, data, word2id_map, id2word_map, word2freq_map):
    """
        Takes a list of text samples, and then generates word2id_map, id2word_map and the freq map.
        Input:
            data = ["text1 is here", "text2 is here"]
        Output:
            word2id_map: {'word1': 1, 'word2': 2 .... }
            id2word_map: {1: 'word1', 2: 'word2' .... }
            freq_map: {'word1': 20, 'word2': 113 .... }
    """
    vocab_index = 1
    for comment in data:
        words = tokenizer.tokenize(comment)
        words = [''.join([c.lower() for c in word if c not in PUNCTUATIONS]) for word in words]
        words = [word for word in words if word not in STOPWORDS]
     
        for word in words:
            if word not in word2id_map:
                word2id_map[word] = vocab_index
                id2word_map[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1                                                                                                                               
            word2freq_map[word] += 1
    word2id_map['UNK'] = len(word2id_map) + 1
    id2word_map[len(word2id_map)] = 'UNK'
    # return word2id_map, id2word_map, word2freq_map


def _convert_sample_to_vector_from_scratch(text, word2id_map):
    """ Takes a sample and cleans it up
        Then converts it into a 2-d vector of word-embeddings.
    """
    text = tokenize(text)
    seq, _emb = [], []
    for w in text:
        # Convert to simple characters, or UNI-DECODE 'em
        try:
            w = unidecode(w)
        except:
            pdb.set_trace()
            print "ERROR: unidecode(%s)" %(w)
            w = w.decode("utf-8")
      
        w = w.lower()
            
        # Skip stopwords & punctuations & URLs
        if w in STOPWORDS or w in PUNCTUATIONS:
            continue
      
        # Skip numbers
        #if REGEX_numbers.match(w):
        #    continue
      
        # Skip words which have length less than this minimum length
        #if len(w) < MIN_LENGTH_OF_KEYWORD: 
        #    continue
      
        seq.append(word2id_map.get(w, word2id_map['UNK']))
    return seq




def transform(texts, word2id_map):
    X = []
    for i, text in enumerate(texts):
        #X.append(_convert_sample_to_vector_from_scratch(text))
        X.append(_convert_sample_to_vector_from_scratch(text, word2id_map))                                                                           
    return X


def get_raw_dataset():
    
    Xtrain, Ytrain, Xtest, Ytest = [], [], [], [] 
    normal='normalnews.json'
    weird='weirdnews.json'
    
    docs = []
    docs = extr.Read_Store(normal, docs)
    docs = extr.Read_Store(weird, docs)
    shuffle(docs)
    
    read = -1
    #read = 100
    
    if read > 0:
        docs = docs[:read]
    else:
        read = len(docs)

    train_count = int(0.8 * read)
    trains = docs[:train_count]
    tests = docs[train_count:]

    Xtrain = [ele[0] for ele in trains]
    Ytrain = [ele[1] for ele in trains]
    Xtest = [ele[0] for ele in tests]
    Ytest = [ele[1] for ele in tests]

    return Xtrain, Ytrain, Xtest, Ytest



def convert_to_tfidf(Xtrain, Ytrain, Xtest, Ytest):
    print "Encoder Type: tfidf"
    #tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = TfidfVectorizer(min_df=3,max_features=None, 
                             strip_accents='unicode',analyzer='word',
                             token_pattern=r'\w{1,}',ngram_range=(1,2),
                             use_idf=1,smooth_idf=1,stop_words='english',
                             )

    
    Traindata = tfidf_vectorizer.fit_transform(Xtrain)
    testdata = tfidf_vectorizer.transform(Xtest)
    
    Traindata=Traindata.todense()
    testdata=testdata.todense()
    
    return Traindata, Ytrain, testdata, Ytest



def convert_to_word_embeddings(Xtrain, Ytrain, Xtest, Ytest):
    print "Encoder Type: word-embneddings"
    tokenizer = MosesTokenizer()
    word2id_map, id2word_map, word2freq_map = {}, {}, defaultdict(int)
    _gen_vocab(tokenizer, Xtrain, word2id_map, id2word_map, word2freq_map)
    
    Traindata = transform(Xtrain, word2id_map)
    testdata = transform(Xtest, word2id_map)

    Traindata = pad_sentences(Traindata, word2id_map)
    testdata = pad_sentences(testdata, word2id_map)

    Ytrain = keras.utils.np_utils.to_categorical(Ytrain)
    Ytest = keras.utils.np_utils.to_categorical(Ytest)
    
    #pdb.set_trace()
    return Traindata, Ytrain, testdata, Ytest, word2id_map, id2word_map

# X= np.loadtxt("traindata.txt")
# data=len(X[0])
# Traindata=X[:,np.r_[0,1:data-1]]
# TrainLabels=X[:,data-1]
# testdata=np.loadtxt("testdata.txt")
# testlabels=np.loadtxt("testlabels.txt")


'''

nb=naivebayes
svm=support vectors machines
rfc=randomforestclassifiers
gbc=graidentboostingclassifiers
abc=adaboostingclassifiers
nn = neural networks
dt=decisiontreeclassifier

'''
# pprint(Testdoc)
# pprint(testdata)

if __name__=="__main__":
    Traindata, TrainLabels, testdata, testlabels = get_raw_dataset()
    
    if sys.argv[1]=="nb":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.Gaussian_NB(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="lr":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.lr(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="svm":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.SVM_predict(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="rfc":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.randomforest_predict(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="gbc":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.XGBoost(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="abc":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.ADABoost(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="nn":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.NN(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="dt":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.Decision_tree(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="automl":
        Traindata, TrainLabels, testdata, testlabels = convert_to_tfidf(Traindata, TrainLabels, testdata, testlabels)
    	clf.autoML(Traindata,TrainLabels,testdata,testlabels)
    if sys.argv[1]=="lstm":
        Traindata, TrainLabels, testdata, testlabels, word2id_map, id2word_map = convert_to_word_embeddings(Traindata, TrainLabels, testdata, testlabels)
	W = get_embedding_weights("word2vec", word2id_map)
    	dl_models.LSTM_train(Traindata,TrainLabels,testdata,testlabels, word2id_map, W)
    if sys.argv[1]=="cnn":
        Traindata, TrainLabels, testdata, testlabels, word2id_map, id2word_map = convert_to_word_embeddings(Traindata, TrainLabels, testdata, testlabels)
	W = get_embedding_weights("word2vec", word2id_map)
    	dl_models.CNN_train(Traindata,TrainLabels,testdata,testlabels, word2id_map, W)


