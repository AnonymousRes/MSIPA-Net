import tensorflow as tf
import numpy as np
import os
import keras.backend as K
import pandas as pd
import keras
import datetime
from sklearn import metrics
from sklearn.metrics import jaccard_score, hamming_loss, label_ranking_average_precision_score, f1_score, roc_auc_score,auc,precision_recall_curve
from sklearn.model_selection import *
from keras.preprocessing import sequence
from keras.layers import *
from keras import *
from Transformer import *
from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects
import math
from typing import Union, Callable, Optional
from keras.layers import Layer, Add, Dropout
import keras.activations
from keras import initializers
from keras import backend as K
from keras.utils import get_custom_objects


import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
test_result = []
results = []
result = []


def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]),dtype=int)
    b[np.arange(len(a)), a] = 1
    return b


def my_test(y_true, y_pred):
    y_true_after = []
    y_pred_after = []
    # y_true = np.array(y_true, dtype=int)
    sumequal0 = 0
    for ptrue, ppred in zip(y_true, y_pred):
        for dtrue, dpred in zip(ptrue, ppred):
            if sum(dtrue) > 0.0 or sum(dtrue) > 0:
                y_true_after.append(dtrue)
                y_pred_after.append(dpred)
    y_true_after = np.array(y_true_after)
    y_pred_after = np.array(y_pred_after)
    yp = props_to_onehot(y_pred_after)
    auroc = tf.keras.metrics.AUC(num_thresholds=100, curve='ROC', multi_label=False)
    auprc = tf.keras.metrics.AUC(num_thresholds=100, curve='PR', multi_label=False)
    auroc.update_state(y_true_after, y_pred_after)
    auprc.update_state(y_true_after, y_pred_after)
    my_auroc = auroc.result().numpy()
    my_auprc = auprc.result().numpy()
    my_micro_f1 = f1_score(y_true_after, yp, average='micro', zero_division=0)
    my_macro_f1 = f1_score(y_true_after, yp, average='macro', zero_division=0)
    print([
        round(my_micro_f1, 6),
        round(my_macro_f1, 6),
        round(my_auroc, 6),
        round(my_auprc, 6),
    ])
    result.append([
        round(my_micro_f1, 6),
        round(my_macro_f1, 6),
        round(my_auroc, 6),
        round(my_auprc, 6),
    ])


def get_msipa(seqlen_, max_features_, rnnlay_units_, dlay_units_):
    inputs = Input(shape=(seqlen_, max_features_))
    embeddings = Dense(rnnlay_units_, activation=None, use_bias=False)(inputs)
    convs = keras.layers.Conv1D(filters=dlay_units_, kernel_size=3, dilation_rate=2, strides=1, padding='same')(embeddings)
    convm = keras.layers.Conv1D(filters=dlay_units_, kernel_size=3, dilation_rate=3, strides=1, padding='same')(embeddings)
    convl = keras.layers.Conv1D(filters=dlay_units_, kernel_size=3, dilation_rate=4, strides=1, padding='same')(embeddings)

    gout = keras.layers.GRU(rnnlay_units_, return_sequences=True, dropout=0.5)(embeddings)
    sgout = keras.layers.GRU(rnnlay_units_, return_sequences=True, dropout=0.5)(convs)
    mgout = keras.layers.GRU(rnnlay_units_, return_sequences=True, dropout=0.5)(convm)
    lgout = keras.layers.GRU(rnnlay_units_, return_sequences=True, dropout=0.5)(convl)

    shortp = ScaledDotProductAttention(False, False)([sgout, sgout, gout])
    mediap = ScaledDotProductAttention(False, False)([mgout, mgout, gout])
    longp = ScaledDotProductAttention(False, False)([lgout, lgout, gout])

    tinput = keras.layers.Concatenate()([shortp, mediap])
    tinput = keras.layers.Concatenate()([tinput, longp])

    encoding = PositionEncoding(rnnlay_units_)(embeddings)
    encoding = PAdd()([embeddings, encoding])
    hout = TransformerBlock(name='TB1', num_heads=dlay_units_, residual_dropout=0.5)(encoding)
    hout = TransformerBlock(name='TB2', num_heads=dlay_units_, residual_dropout=0.5)(hout)
    hout = TransformerBlock(name='TB3', num_heads=dlay_units_, residual_dropout=0.5)(hout)
    hout = TransformerBlock(name='TB4', num_heads=dlay_units_, residual_dropout=0.5)(hout)

    hout = keras.layers.Concatenate()([hout, tinput])
    hout = LayerNormalization()(hout)

    hout_final = Dense(36, activation=keras.activations.softmax)(hout)
    model = Model(inputs=inputs, outputs=hout_final, name='MSIPA')
    return model


def train_model(model_, x_train_, y_train_, x_test_, y_test_, batch_size_, epochs_, patience_, verbose_):
    print(model_.name + ':')
    model_.compile('adam', loss=keras.losses.categorical_crossentropy, metrics=keras.metrics.categorical_accuracy)
    # model_.summary()
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                              patience=patience_,
                                              restore_best_weights=True)
    model_.fit(x_train_, y_train_,
               batch_size=batch_size_,
               epochs=epochs_,
               # validation_split=0.2,
               validation_data=(x_test_, y_test_),
               callbacks=[earlystop],
               verbose=verbose_
               )
    model_.save_weights('/home/XXX/icu_data/%s_WA.h5' % model_.name)
    y_pred_ = model_.predict(x_test_)
    my_test(y_true=y_test_, y_pred=y_pred_)

if __name__ == '__main__':
    batch_size = 256
    seqlen = 35
    epochs = 1000
    patience = 20
    verbose = 0
    rnnlay_units = 64
    dlay_units = 64
    print('Loading data...')
    x = np.load('/home/xxx/icu_data/x_ward.npy', allow_pickle=True)
    y = np.load('/home/xxx/icu_data/y_ward.npy', allow_pickle=True)
    max_features = x[0].shape[1]
    results = []
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(x):
        x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]
        x_train = sequence.pad_sequences(x_train, maxlen=seqlen, padding='post')
        x_test = sequence.pad_sequences(x_test, maxlen=seqlen, padding='post')
        y_train = sequence.pad_sequences(y_train, maxlen=seqlen, padding='post')
        y_test = sequence.pad_sequences(y_test, maxlen=seqlen, padding='post')
        result = []
        print('\tMicro F1\t\t\tMacro F1\t\t\tAUROC\t\t\tAUPRC')
        train_model(get_msipa(seqlen, max_features, rnnlay_units, dlay_units),
                                  x_train, y_train, x_test, y_test, batch_size, epochs, patience, verbose)
        # exit(0)
        npresult = np.array(result)
        print(npresult)
        results.append(npresult)
        # exit(0)
    print('\n\n\n\n\n')
    npresults = np.array(results)
    print(npresults.shape)
    npresults = np.swapaxes(npresults, 0, 1)
    print(npresults.shape)
    # np.save('/home/XXX/icu_data/baseline_result.npy', naresults)
    # naresults = np.load('/home/XXX/icu_data/baseline_result.npy', allow_pickle=True)
    # print(naresults)
    np.set_printoptions(precision=4)
    print(npresults.mean(axis=1))
    np.set_printoptions(precision=4)
    print(npresults.std(axis=1))
