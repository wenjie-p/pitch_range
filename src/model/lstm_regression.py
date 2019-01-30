#!/usr/bin/env python
import os
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import Callback, ModelCheckpoint
import codecs

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

    def __init__(self, data, batch_size, num_steps, input_dim, batches):
        self.data = data
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.cur_idx = 0
        self.batches = batches

    def generate_test(self):

        scaler = MinMaxScaler(feature_range=(0,1))

        batches = len(self.data)//(self.num_steps+1)
        
        xx = np.zeros((self.batch_size, self.num_steps, self.input_dim))
        yy = np.zeros((self.batch_size))

        while self.cur_idx < self.batches:

            for i in range(self.batch_size):
                xt = self.data[self.cur_idx: self.cur_idx+self.num_steps]
                xt = scaler.fit_transform(xt)
                self.cur_idx += self.num_steps
                xx[i,:,:] = xt
                yy[i] = self.data[self.cur_idx][0]
                self.cur_idx += 1
                if self.cur_idx >= self.batches:
                    return
                
            yield xx, yy

    def generate_train(self):

        scaler = MinMaxScaler(feature_range=(0,1))

        xx = np.zeros((self.batch_size, self.num_steps, self.input_dim))
        yy = np.zeros((self.batch_size))
        
        while True:
            for i in range(self.batch_size):
                if self.cur_idx >= len(self.data):
                    self.cur_idx = 0
                xt = self.data[self.cur_idx: self.cur_idx+self.num_steps]
                xt = scaler.fit_transform(xt)
                xx[i,:,:] = xt
                self.cur_idx += self.num_steps
                yy[i] = self.data[self.cur_idx][0]
                self.cur_idx += 1
            yield xx, yy

def init_lstm(num_steps, input_dim, hidden_size = 100):

    model = Sequential()
    model.add(LSTM(hidden_size, input_shape = (num_steps, input_dim), return_sequences = True))
    model.add(LSTM(hidden_size, input_shape = (num_steps, input_dim), return_sequences = False))
    model.add(Dense(1))
    model.add(Activation("linear"))
    optimizer = Adam
    model.compile(loss="mean_squared_error", optimizer = "adam")

    print(model.summary())

    return model

def analyze_data(fp, batch_size, num_steps):
    
    data = np.loadtxt(fp, "float")
    length = len(data)
    batches = length//(num_steps+1)
    print("batches is : {}".format(batches))
    spe = batches // batch_size

    return data, batches, spe

def model_fit(fp, batch_size, num_steps, input_dim):

    data, batches, spe = analyze_data(fp, batch_size, num_steps)
    
    print(spe)
    train_data_generator = KerasBatchGenerator(data, batch_size, num_steps, input_dim, batches)
    md = init_lstm(num_steps, input_dim, 100)
    #checkpoint = ModelCheckpoint(filepath = "./", monitor = "val_acc", verbose = 1, save_best_only = True, mode = "max")
    #callback_list = [checkpoint]

    md.fit_generator(train_data_generator.generate_train(), steps_per_epoch = spe, epochs = 20)
    #md.fit_generator(train_data_generator.generate(), steps_per_epoch = spe, callbacks = callback_list, epochs = 30)
    md.save_weights("latest_final_model_weights.h5")


def calculate_loss(gen, model, steps):

    loss = []
    count = 0
    for i in range(steps):
        x, y = next(gen.generate_train())
        yhat = model.predict(x)
        mae = sum([abs(yhat[i]-y[i]) for i in range(len(y))])/len(y)
        loss.append(mae)
        count += len(y)
    ave = np.array(loss).mean()
    return ave, count

def model_evaluate(train, test,  batch_size, num_steps, input_dim):

    train_data, train_batches, train_spe = analyze_data(train, batch_size, num_steps)
    test_data, test_batches, test_spe =  analyze_data(test, batch_size, num_steps)

    md = init_lstm(num_steps, input_dim, 100)
    md.load_weights("latest_final_model_weights.h5")

    train_data_generator = KerasBatchGenerator(train_data, batch_size, num_steps, input_dim, train_batches)
    train_ave, train_co = calculate_loss(train_data_generator, md, train_spe)
    print("The training steps: {} with mae: {}".format(train_co, train_ave))

    test_data_generator = KerasBatchGenerator(test_data, batch_size, num_steps, input_dim, test_batches)
    test_ave, test_co = calculate_loss(test_data_generator, md, test_spe)
    print("The test  steps: {} with mae: {}".format(test_co, test_ave))
    

def main(feats, stage):

    batch_size, num_steps, input_dim = load_cfg("../conf/nn.conf")
    if stage == "0":
        print("Training starts...")
        fp = feats+"/"+"train"
        model_fit(fp, batch_size, num_steps, input_dim)
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

