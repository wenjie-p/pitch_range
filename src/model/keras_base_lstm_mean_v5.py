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
import h5py

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
                stds = set([e[-1] for e in samples])
                means = set([e[-2] for e in samples])
                if len(means) > 1 or len(stds) > 1:
                    continue
                xt = [e[:self.input_dim] for e in samples]
                #xt = scaler.fit_transform(xt)
                xx[i,:,:] = xt
                yt = samples[-1][-2]
                yy[i] = yt
                i+=1

            yield xx, yy


def init_lstm(loss, num_steps, input_dim, hidden_size = 50):

    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences = True, input_shape = (num_steps, input_dim)))
    #model.add(Bidirectional(LSTM(hidden_size, input_shape = (num_steps, input_dim), return_sequences = False)))
    model.add(LSTM(hidden_size, return_sequences = True ))
    model.add(LSTM(hidden_size, return_sequences = False ))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss = loss, optimizer = "sgd")


    print(model.summary())

    return model

def analyze_data(fp, batch_size, num_steps):
    
    h5f = h5py.File(fp, "r")
    data = h5f["data_set"][:]
    h5f.close()
    #data = np.loadtxt(fp, "float")
    length = len(data)
    batches = length//(num_steps+1)
    spe = batches // batch_size

    return data, spe

def model_fit(ftrain, fdev, batch_size, num_steps, input_dim, fmd):
    
    loss = "mean_absolute_error"
    md = init_lstm(loss, num_steps, input_dim, 100)
    es = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 10)
    mc = ModelCheckpoint(filepath = "./base_mean_mds/"+fmd, mode = "min", save_best_only = True, verbose = 1)
    now = datetime.now()
    timestr = now.strftime("%Y-%m-%d %H:%M")
    tb = TensorBoard(log_dir = "./base_mean_logs/{}".format(timestr), batch_size = batch_size)

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

def get_loss(fout, gen, spe, md):

    val = 0
    co = 0
    fp = codecs.open(fout, "w", encoding = "utf8")

    for i in range(spe):
        x, y = next(gen.generate_train())
        yhat = md.predict(x)
        val += sum([abs(yhat[j] - y[j])/y[j] for j in range(len(y))]) 
        co += len(y)

        yhat = [str(yy) for yy in yhat.tolist()]
        y = [str(yy) for yy in y.tolist()]
        if len(yhat) != len(y):
            print("Error")
            exit(0)
        for j in range(len(yhat)):
            fp.write(" ".join([y[j], yhat[j]])+"\n")

    fp.close()

    val = val/co*100
    print("The loss is: {}%".format(val))

def model_evaluate(train, dev, test,  batch_size, num_steps, input_dim, fmd):

    train_data, train_spe = analyze_data(train, batch_size, num_steps)
    dev_data, dev_spe = analyze_data(dev, batch_size, num_steps)
    test_data, test_spe =  analyze_data(test, batch_size, num_steps)
    
    loss = "mean_absolute_percentage_error"
    md = init_lstm(loss, num_steps, input_dim, 100)
    md.load_weights("./base_mean_mds/"+fmd)
    
    train_data_generator = KerasBatchGenerator(train_data, batch_size, num_steps, input_dim)
    dev_data_generator = KerasBatchGenerator(dev_data, batch_size, num_steps, input_dim)
    test_data_generator = KerasBatchGenerator(test_data, batch_size, num_steps, input_dim)

    ftrain = "./train.out"
    fdev = "./dev.out"
    ftest = "./test.out"

#    get_loss(ftrain, train_data_generator, train_spe, md)
#    get_loss(fdev, dev_data_generator, dev_spe, md)
#    get_loss(ftest, test_data_generator, test_spe, md)
#    print("{} {} {}".format(train_spe, dev_spe, test_spe))
    
#    return 0

    train_loss = md.evaluate_generator(train_data_generator.generate_train(), steps = train_spe, verbose = 0)
    dev_loss = md.evaluate_generator(dev_data_generator.generate_train(), steps = dev_spe, verbose = 0)
    test_loss = md.evaluate_generator(test_data_generator.generate_train(), steps = test_spe, verbose = 0)
    print("The train loss is: {: >.5f}".format(train_loss))
    print("The dev loss is: {: >.5f}".format(dev_loss))
    print("The test loss is: {: >.5f}".format(test_loss))
    

def main(feats, stage):

    #batch_size, num_steps, input_dim = load_cfg("../conf/nn.conf")
    batch_size   = 50
    num_steps    = 30
    input_dim    = 40
    fmd          = "model_v5.h5"
    if stage == "0":
        print("Training starts...")
        ftrain = feats+"/"+"train.h5"
        fdev = feats+"/"+"dev.h5"
        model_fit(ftrain, fdev,  batch_size, num_steps, input_dim, fmd)
    elif stage == "1":
        print("Evaluation starts...")
        train = feats+"/"+"train.h5"
        dev = feats+"/"+"dev.h5"
        test = feats+"/"+"test.h5"
        model_evaluate(train, dev, test, batch_size, num_steps, input_dim, fmd)
    else:
        print("Invalid command.")
        exit(0)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: {} input stage<0: training, 1: test>".format(sys.argv[0]))
        exit(0)
    main(sys.argv[1], sys.argv[2])

