from keras.layers import Embedding, Input, LSTM, Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.models import Sequential, Model
import numpy as np

from sklearn.metrics import classification_report
import pdb



EMBEDDING_DIM = 300
#TRAINABLE_EMBEDDING = True
TRAINABLE_EMBEDDING = False

def lstm_model(sequence_length, EMBEDDING_DIM, embedding_matrix, vocab):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    model.add(Embedding(len(vocab)+1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=TRAINABLE_EMBEDDING, input_length=sequence_length))
    #model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(300))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])
    print model.summary()
    return model


def cnn_model(sequence_length, EMBEDDING_DIM, embedding_matrix, vocab):
#def baseline_CNN(sequences_length_for_training, embedding_dim, embedding_matrix, vocab_size):
    main_input = Input(shape=(sequence_length,), dtype='float32', name='main-input')
    main_input_embedder = Embedding(len(vocab) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=sequence_length, init='uniform')
    embedded_input_main = main_input_embedder(main_input)
 
    convsM = []
    ngram_filters = [2, 3, 4, 5]
    conv_hidden_units = [200, 200, 200, 200]
    for n_gram, hidden_units in zip(ngram_filters, conv_hidden_units):
        conv_layer = Convolution1D(nb_filter=hidden_units,
                             filter_length=n_gram,
                             border_mode='same',
                             #border_mode='valid',
                             activation='tanh', name='Convolution-'+str(n_gram)+"gram")
        mid = conv_layer(embedded_input_main)
 
        # Use Flatten() instead of MaxPooling()
        #flat_M = TimeDistributed(Flatten(), name='TD-flatten-mid-'+str(n_gram)+"gram")(mid)
        #convsM.append(flat_M)
 
        # Use GlobalMaxPooling1D() instead of Flatten()
        pool_M = GlobalMaxPooling1D()(mid)
        convsM.append(pool_M)
 
    convoluted_mid = Merge(mode='concat')(convsM)
    CONV_DIM = sum(conv_hidden_units)
    encode_mid = Dense(300, name='dense-intermediate-mid-encoder')(convoluted_mid)
    encode_mid_drop = Dropout(0.2)(encode_mid)
 
    decoded = Dense(300, name='decoded')(encode_mid_drop)
    decoded_drop = Dropout(0.3, name='decoded_drop')(decoded)
    
    output = Dense(2, activation='sigmoid')(decoded_drop)
    model = Model(input=[main_input], output=output)
    #model.layers[1].trainable = TRAINABLE_EMBEDDINGS
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'recall'])
    model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy', 'recall'])
    print model.summary(line_length=150, positions=[.46, .65, .77, 1.])
    return model


def LSTM_train(Xtrain, Ytrain, Xtest, Ytest, word2id_map, W):
    model = (len(Xtrain[0]), EMBEDDING_DIM, W, word2id_map)
    model.fit(Xtrain, Ytrain, batch_size=32)
    output = model.predict(Xtest)

    testlabels = np.argmax(Ytest, axis=1)
    output = np.argmax(output, axis=1)

    print classification_report(testlabels, output, target_names=['class0', 'class1'])

    pdb.set_trace()


def CNN_train(Xtrain, Ytrain, Xtest, Ytest, word2id_map, W):
    model = cnn_model(len(Xtrain[0]), EMBEDDING_DIM, W, word2id_map)
    model.fit(Xtrain, Ytrain, batch_size=32)
    output = model.predict(Xtest)

    testlabels = np.argmax(Ytest, axis=1)
    output = np.argmax(output, axis=1)

    print classification_report(testlabels, output, target_names=['class0', 'class1'])

    pdb.set_trace()

