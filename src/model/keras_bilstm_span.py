#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
from datetime import datetime
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, LSTM, Bidirectional
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
import codecs
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction=0.5
#set_session(tf.Session(config = config))

np.random.seed(42)

def load_cfg(cfg):
    # TO
    # We should use argparser instead hard code

    with codecs.open(cfg, "r", encoding = "utf8") as f:
        data = f.readlines()
        params = {}
        for line in data:
            line = line.strip().split("=")
            params[line[0]] = int(line[1])
        
        return params["batch_size"], params["num_steps"], params["input_dim"]

class KerasBatchGenerator(object):

    def __init__(self, data, batch_size, num_steps, input_dim):
        self.data = data
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.cur_idx = 0
        self.skip_steps = 5

    def generate_test(self):

        scaler = MinMaxScaler(feature_range=(0,1))

        batches = len(self.data)//(self.num_steps+1)

        for i in range(batches):
            xt = self.data[self.cur_idx: self.cur_idx+self.num_steps]
            xt = scaler.fit_transform(xt)
            self.cur_idx += self.num_steps
            yt = self.data[self.cur_idx][0]
            self.cur_idx += 1

            yield xt, yt

    def generate_train(self):

        #scaler = MinMaxScaler(feature_range=(0,1))

        xx = np.zeros((self.batch_size, self.num_steps, self.input_dim))
        yy = np.zeros((self.batch_size))
        
        while True:
            i = 0
            while i < self.batch_size:
                if self.cur_idx + self.num_steps >= len(self.data):
                    self.cur_idx = 0
                samples = self.data[self.cur_idx: self.cur_idx+self.num_steps]
                self.cur_idx += self.skip_steps
                uttids = set([e[-3] for e in samples])
                if len(uttids) > 1:
                    continue
                xt = [e[:self.input_dim] for e in samples]
                #xt = scaler.fit_transform(xt)
                xx[i,:,:] = xt
                # -1 means span
                yt = [e[-1] for e in samples]
                yy[i] = self.data[self.cur_idx][0]
                i+=1

            yield xx, yy

def init_lstm(num_steps, input_dim, hidden_size = 50):

    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_size, return_sequences = True ), input_shape = (num_steps, input_dim)))
    #model.add(Bidirectional(LSTM(hidden_size, input_shape = (num_steps, input_dim), return_sequences = False)))
    model.add(Bidirectional(LSTM(hidden_size, return_sequences = True )))
    model.add(Bidirectional(LSTM(hidden_size, return_sequences = False )))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss="mean_absolute_error", optimizer = "sgd")


    print(model.summary())

    return model

def analyze_data(fp, batch_size, num_steps):
    
    data = np.loadtxt(fp, "float")
    length = len(data)
    batches = length//(num_steps+1)
    spe = batches // batch_size

    return data, spe

def model_fit(ftrain, fdev, batch_size, num_steps, input_dim):
    
    md = init_lstm(num_steps, input_dim, 50)
    es = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 10)
    mc = ModelCheckpoint(filepath = "./span_best_model_v1.h5", mode = "min", save_best_only = True, verbose = 1)
    now = datetime.now()
    timestr = now.strftime("%Y-%m-%d %H:%M")
    tb = TensorBoard(log_dir = "./logs/{}".format(timestr), batch_size = batch_size)

    train_data, train_spe = analyze_data(ftrain, batch_size, num_steps)
    train_data_generator = KerasBatchGenerator(train_data, batch_size, num_steps, input_dim)
    #checkpoint = ModelCheckpoint(filepath = "./", monitor = "val_acc", verbose = 1, save_best_only = True, mode = "max")
    #callback_list = [checkpoint]
    
    dev_data, dev_spe = analyze_data(fdev, batch_size, num_steps)
    dev_data_generator = KerasBatchGenerator(dev_data, batch_size, num_steps, input_dim)


    md.fit_generator(train_data_generator.generate_train(), 
            steps_per_epoch = train_spe, 
            validation_data = dev_data_generator.generate_train(),
            validation_steps = dev_spe,
            callbacks = [es, mc, tb],
            epochs = 100)

def get_loss(gen, spe, md):

    val = 0
    co = 0
    for i in range(spe):
        x, y = next(gen.generate_train())
        yhat = md.predict(x)
        val += sum([abs(yhat[j] - y[j])/y[j] for j in range(len(y))]) 
        co += len(y)

    val = val/co*100
    print("The loss is: {}%".format(val))

def model_evaluate(train, test,  batch_size, num_steps, input_dim):

    train_data, train_spe = analyze_data(train, batch_size, num_steps)
    test_data, test_spe =  analyze_data(test, batch_size, num_steps)

    md = init_lstm(num_steps, input_dim, 50)
    md.load_weights("span_best_model_v1.h5")
    
    train_data_generator = KerasBatchGenerator(train_data, batch_size, num_steps, input_dim)
    test_data_generator = KerasBatchGenerator(test_data, batch_size, num_steps, input_dim)

    get_loss(train_data_generator, train_spe, md)
    get_loss(test_data_generator, test_spe, md)
    
    return 0

    train_loss = md.evaluate_generator(train_data_generator.generate_train(), steps = train_spe, verbose = 0)
    test_loss = md.evaluate_generator(test_data_generator.generate_train(), steps = test_spe, verbose = 0)
    print("The train loss is: {: >.5f}".format(train_loss))
    print("The test loss is: {: >.5f}".format(test_loss))
    

def main(feats, stage):

    batch_size, num_steps, input_dim = load_cfg("../conf/nn.conf")
    if stage == "0":
        print("Training starts...")
        ftrain = feats+"/"+"train"
        fdev = feats+"/"+"dev"
        model_fit(ftrain, fdev, batch_size, num_steps, input_dim)
    elif stage == "1":
        print("Evaluation starts...")
        train = feats+"/"+"train"
        test = feats+"/"+"test"
        model_evaluate(train, test, batch_size, num_steps, input_dim)
    else:
        print("Invalid command.")
        exit(0)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: {} input stage<0: training, 1: test>".format(sys.argv[0]))
        exit(0)
    main(sys.argv[1], sys.argv[2])

