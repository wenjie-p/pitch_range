#!/usr/bin/env python
import os
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, LSTM
from keras.optimizers import SGD
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

    def __init__(self, data, batch_size, num_steps, input_dim):
        self.data = data
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.cur_idx = 0

    def generate(self):

        xx = np.zeros((self.batch_size, self.num_steps, self.input_dim))
        yy = np.zeros((self.batch_size))
        
        while True:
            for i in range(self.batch_size):
                if self.cur_idx >= len(self.data):
                    self.cur_idx = 0
                utt_id = self.data[self.cur_idx][-3]
                j = 0
                x = []
                y = 0
                while self.data[self.cur_idx+j][-3] == utt_id:
                    xt = self.data[self.cur_idx+j][:-3]
                    x.append(xt)

                    y = slef.data[self.cur_idx+j][-2]
                    j+=1

                xx[i,:,:] = xt
                yy[i] = y
                self.cur_idx += j

            yield xx, yy

def init_lstm(num_steps, input_dim, hidden_size = 50):

    model = Sequential()
    model.add(LSTM(hidden_size, input_shape = (None, input_dim), return_sequences = True))
    model.add(LSTM(hidden_size, return_sequences = False))
    model.add(Dense(1))
    model.add(Activation("linear"))
    sgd = SGD(lr = 0.01, decay = 1e-4, momentum = 0.9, nesterov = True)
    model.compile(loss="mean_absolute_error", optimizer = sgd)

    print(model.summary())

    return model

def analyze_data(data, num_steps):

    length = len(data)

    return length//(num_steps+1)

def train(fp, batch_size, num_steps, input_dim):
    
    md = init_lstm(num_steps, input_dim, 50)

    data = np.loadtxt(fp, "float")
    print(len(data))
    samples_tot = analyze_data(data, num_steps)
    print(samples_tot)
    spe = samples_tot//batch_size
    print(spe)
    train_data_generator = KerasBatchGenerator(data, batch_size, num_steps, input_dim)

    #checkpoint = ModelCheckpoint(filepath = "./", monitor = "val_acc", verbose = 1, save_best_only = True, mode = "max")
    #callback_list = [checkpoint]

    md.fit_generator(train_data_generator.generate(), steps_per_epoch = spe, epochs = 20)
    #md.fit_generator(train_data_generator.generate(), steps_per_epoch = spe, callbacks = callback_list, epochs = 30)
    md.save_weights("sgd_final_model_weights.h5")

def calculate_loss(gen, model, steps):

    loss = []
    count = 0
    for i in range(steps):
        x, y = next(gen.generate())
        yhat = model.predict(x)
        mae = sum([abs(yhat[i]-y[i]) for i in range(len(y))])/len(y)
        loss.append(mae)
        count += len(y)
    ave = np.array(loss).mean()
    return ave, count


def test(fp, batch_size, num_steps, input_dim):

    md = init_lstm(num_steps, input_dim, 50)
    md.load_weights("sgd_final_model_weights.h5")

    data = np.loadtxt(fp, "float")
    samples_tot = analyze_data(data, num_steps)
    print(samples_tot)
    spe = samples_tot//batch_size
    print(spe)
    generator = KerasBatchGenerator(data, batch_size, num_steps, input_dim)
    #loss = md.evaluate_generator(generator.generate(), steps = spe )
    loss, count = calculate_loss(generator, md, spe)
    print("The mae is: {} with {} samples".format(loss, count))

def main(feats, stage):

    batch_size, num_steps, input_dim = load_cfg("../conf/nn.conf")
    if stage == "0":
        print("Training starts...")
        fp = feats+"/"+"dev"
        train(fp, batch_size, num_steps, input_dim)
    elif stage == "1":
        print("Testing starts...")
        fp = feats+"/"+"test"
        test(fp, batch_size, num_steps, input_dim)
    else:
        print("Invalid command.")
        exit(0)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: {} input stage<0: training, 1: test>".format(sys.argv[0]))
        exit(0)
    main(sys.argv[1], sys.argv[2])
