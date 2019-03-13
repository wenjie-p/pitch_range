#!/usr/bin/env python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import sys
from datetime import datetime
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, LSTM, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
import codecs
import random
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction=0.5
#set_session(tf.Session(config = config))

np.random.seed(42)

# This function has been deprecated.
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


## Data generator
class KerasBatchGenerator(object):

    def __init__(self, fp, batch_size, num_steps, input_dim, op):

        self.fp = fp
        self.data = self.load_data()
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.op = op
        self.cur_idx = 0
        self.tot = len(self.data)
        self.idxs = list(range(self.tot))

    def get_spe(self):

        return self.tot//(self.num_steps*self.batch_size)

    def load_data(self):

        data = np.loadtxt(self.fp, "float")
        return data

    def generate(self):

        #scaler = MinMaxScaler(feature_range=(0,1))

        xx = np.zeros((self.batch_size, self.num_steps, self.input_dim))
        yy = np.zeros((self.batch_size))

        random.shuffle(self.idxs)

        while True:

            for i in range(self.batch_size):
                x, y = "", ""
                while True:
                    if self.cur_idx+self.num_steps > self.tot:
                        self.cur_idx = 0
                        random.shuffle(self.idxs)
                    beg = self.idxs[self.cur_idx]
                    end = beg+self.num_steps
                    if end > self.tot:
                        self.cur_idx += 1
                        continue
                    samples = self.data[beg:end]
                    means = set(samples[:,-2])
                    stds  = set(samples[:,-1])
                    if len(means) > 1 or len(stds) > 1:
                        self.cur_idx += 1
                    else:
                        x = samples[:,:-2]
                        y = samples[0][self.op]
                        break
                xx[i,:,:] = x
                yy[i] = y
            
                self.cur_idx += 1

            yield xx, yy
                
        

def init_lstm(loss, num_steps, input_dim, hidden_size = 50):

    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences = True, input_shape = (num_steps, input_dim), activation = "relu"))
    #model.add(Bidirectional(LSTM(hidden_size, input_shape = (num_steps, input_dim), return_sequences = False)))
    model.add(LSTM(hidden_size, return_sequences = True, activation = "relu" ))
    model.add(LSTM(hidden_size, return_sequences = False, activation = "relu" ))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss = loss, optimizer = "sgd")


    print(model.summary())

    return model

def create_data(fp, num_steps, idx):

    dataX, dataY = [], []
    data = np.loadtxt(fp, "float")
    cur = 0

    for i in range(len(data) - num_steps - 1):
        samples = data[i:i+num_steps]
        means = set([sample[-2] for sample in samples])
        stds = set([sample[-1] for sample in samples])
        if len(means) > 1 or len(stds) > 1:
            continue
        x = samples[:,:-2]
        y = samples[0][idx]
        dataX.append(x)
        dataY.append(y)

    return np.array(dataX), np.array(dataY)

# This function has been deprecated
def analyze_data(data, batch_size, num_steps):
    
    data = np.loadtxt(fp, "float")
    length = len(data)
    batches = length//num_steps
    spe = batches // batch_size

    return data, spe

def model_fit(ftrain, fdev, batch_size, num_steps, input_dim, hidden_size, fmd, op):
    
    loss = "mean_absolute_error"
    md = init_lstm(loss, num_steps, input_dim, hidden_size)
    es = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 10)
    mc = ModelCheckpoint(filepath = "./base_mean_mds/"+fmd, mode = "min", save_best_only = True, verbose = 1)
    now = datetime.now()
    timestr = now.strftime("%Y-%m-%d %H:%M")
    tb = TensorBoard(log_dir = "./base_mean_logs/{}".format(timestr), batch_size = batch_size)
    
    
    train_data_generator = KerasBatchGenerator(ftrain, batch_size, num_steps, input_dim, op)
    train_spe = train_data_generator.get_spe() 
    
    dev_data_generator = KerasBatchGenerator(fdev, batch_size, num_steps, input_dim, op)
    dev_spe = dev_data_generator.get_spe()

    md.fit_generator(train_data_generator.generate(), 
            steps_per_epoch = train_spe, 
            validation_data = dev_data_generator.generate(),
            validation_steps = dev_spe,
            callbacks = [es, mc, tb],
            epochs = 100)

def evaluate_test_manually(X, Y, md, fout):

    Yhat = md.predict(X)
    fp = codecs.open(fout, "w", encoding = "utf8")

    if len(Y) != len(Yhat):
        print("Yhat size: {} vs Y size: {}".format(len(Yhat), len(Y)))
        exit(0)

    loss = 0
    for i in range(len(X)):
        y = Y[i]
        yhat = Yhat[i][0]
        val = abs((y-yhat)/y)
        loss += val
        line = "{} {}\n".format(y, yhat)
        fp.write(line)

    fp.close()
    mape = loss/len(X)*100

    return mape

def model_evaluate(train, dev, test,  batch_size, num_steps, input_dim, hidden_size, fmd, op):
    
    loss = "mean_absolute_percentage_error"
    md = init_lstm(loss, num_steps, input_dim, hidden_size)
    md.load_weights("./base_mean_mds/"+fmd)

    testX, testY = create_data(test, num_steps, op)
    test_spe = len(testX)//batch_size
    testPredict = md.predict(testX)

    ftest = "./test.out"
    mape = evaluate_test_manually(testX, testY, md, ftest)
    print("The mape of test manually calculated is: {}%".format(mape))

    train_data_generator = KerasBatchGenerator(train, batch_size, num_steps, input_dim, op)
    train_spe = train_data_generator.get_spe()

    dev_data_generator = KerasBatchGenerator(dev, batch_size, num_steps, input_dim, op)
    dev_spe = dev_data_generator.get_spe()

    test_data_generator = KerasBatchGenerator(test, batch_size, num_steps, input_dim, op)
    test_spe = test_data_generator.get_spe()
#    ftrain = "./train.out"
#    fdev = "./dev.out"
    #ftest = "./test.out"

   # get_loss(ftrain, train_data_generator, train_spe, md)
   # get_loss(fdev, dev_data_generator, train_spe, md)
   # get_loss(ftest, test_data_generator, test_spe, md)
    
    #return 0

    train_loss = md.evaluate_generator(train_data_generator.generate(), steps = train_spe, verbose = 0)
    test_loss = md.evaluate_generator(test_data_generator.generate(), steps = test_spe, verbose = 0)
    dev_loss = md.evaluate_generator(dev_data_generator.generate(), steps = dev_spe, verbose = 0)

    print("The train loss is: {: >.5f}".format(train_loss))
    print("The dev loss is: {: >.5f}".format(dev_loss))
    print("The test loss is: {: >.5f}".format(test_loss))
    

def main(feats, stage):

    #batch_size, num_steps, input_dim = load_cfg("../conf/nn.conf")
    batch_size   = 50
    num_steps    = 30
    input_dim    = 40
    hidden_size  = 100
    # op : -1 means span while -2 means level
    op = -2
    fmd          = "model_v4.h5"
    if stage == "0":
        print("Training starts...")
        ftrain = feats+"/"+"train"
        fdev = feats+"/"+"dev"
        model_fit(ftrain, fdev, batch_size, num_steps, input_dim, hidden_size, fmd, op)
    elif stage == "1":
        print("Evaluation starts...")
        train = feats+"/"+"train"
        dev = feats+"/"+"dev"
        test = feats+"/"+"test"
        model_evaluate(train, dev, test, batch_size, num_steps, input_dim, hidden_size,  fmd, op)
    else:
        print("Invalid command.")
        exit(0)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: {} input stage<0: training, 1: test>".format(sys.argv[0]))
        exit(0)
    main(sys.argv[1], sys.argv[2])

