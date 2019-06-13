'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''
import sys
import os
import numpy as np
np.random.seed(123)
from six.moves import cPickle
import json

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam, Adamax, Adadelta, Adagrad, Adagrad, RMSprop, SGD
from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

import time as time_piece
start_time=time_piece.time()

HYPO_DIR = './' + sys.argv[1] + '/'
DATA_DIR = './kitti_data/'

with open(HYPO_DIR + 'hyperparameters.json', 'r') as f:
    params = json.load(f)

save_model = True  # if weights will be saved
weights_file = os.path.join(HYPO_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(HYPO_DIR, 'prednet_kitti_model.json')

# Data files
train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

# Training parameters
nb_epoch = params['nb_epoch']
batch_size = params['batch_size']
samples_per_epoch = params['samples_per_epoch']
N_seq_val = params['N_seq_val']  # number of sequences to use for validation

# Model parameters
nt = 10
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (n_channels, im_height, im_width) if K.image_dim_ordering() == 'th' else (im_height, im_width, n_channels)

#number_of_channels_in_first_layer
nchannels_fl = params['num_filters_first_prednet_layer'] 
layers = params['layers']
if(layers == 6):
    stack_sizes = (n_channels, nchannels_fl, nchannels_fl*2, nchannels_fl*2*2, nchannels_fl*2*2*2, nchannels_fl*2*2*2*2)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (params['A_filter_1'], params['A_filter_2'], params['A_filter_3'], params['A_filter_4'], params['A_filter_5'])
    Ahat_filt_sizes = (params['Ahat_filter_1'], params['Ahat_filter_2'], params['Ahat_filter_3'], params['Ahat_filter_4'], params['Ahat_filter_5'], params['Ahat_filter_6'])
    R_filt_sizes = (params['R_filter_1'], params['R_filter_2'], params['R_filter_3'], params['R_filter_4'], params['R_filter_5'], params['R_filter_6'])
    layer_loss_weights = np.array([1., 0., 0., 0., 0., 0.])
elif(layers == 5):
    stack_sizes = (n_channels, nchannels_fl, nchannels_fl*2, nchannels_fl*2*2, nchannels_fl*2*2*2)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (params['A_filter_1'], params['A_filter_2'], params['A_filter_3'], params['A_filter_4'])
    Ahat_filt_sizes = (params['Ahat_filter_1'], params['Ahat_filter_2'], params['Ahat_filter_3'], params['Ahat_filter_4'], params['Ahat_filter_5'])
    R_filt_sizes = (params['R_filter_1'], params['R_filter_2'], params['R_filter_3'], params['R_filter_4'], params['R_filter_5'])
    layer_loss_weights = np.array([1., 0., 0., 0., 0.])
elif(layers == 4):
    stack_sizes = (n_channels, nchannels_fl, nchannels_fl*2, nchannels_fl*2*2)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (params['A_filter_1'], params['A_filter_2'], params['A_filter_3'])
    Ahat_filt_sizes = (params['Ahat_filter_1'], params['Ahat_filter_2'], params['Ahat_filter_3'], params['Ahat_filter_4'])
    R_filt_sizes = (params['R_filter_1'], params['R_filter_2'], params['R_filter_3'], params['R_filter_4'])
    layer_loss_weights = np.array([1., 0., 0., 0.])
elif(layers == 3):
    stack_sizes = (n_channels, nchannels_fl, nchannels_fl*2)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (params['A_filter_1'], params['A_filter_2'])
    Ahat_filt_sizes = (params['Ahat_filter_1'], params['Ahat_filter_2'], params['Ahat_filter_3'])
    R_filt_sizes = (params['R_filter_1'], params['R_filter_2'], params['R_filter_3'])
    layer_loss_weights = np.array([1., 0., 0])
elif(layers == 2):
    stack_sizes = (n_channels, nchannels_fl)
    R_stack_sizes = stack_sizes
    A_filt_sizes = (params['A_filter_1'],)
    Ahat_filt_sizes = (params['Ahat_filter_1'], params['Ahat_filter_2'])
    R_filt_sizes = (params['R_filter_1'], params['R_filter_2'])
    layer_loss_weights = np.array([1., 0.])
else:
    print("error! layers value: " + str(layers))
    with open (os.path.join(HYPO_DIR, "Error While Training.txt"), "w") as f:
        f.write("Tried to train with " + str(layers) + "but training is only parameterized for maximum 6 layers")
print(stack_sizes)  
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
time_loss_weights[0] = 0

prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True)
inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)
errors_by_time = TimeDistributed(Dense(1, weights=[layer_loss_weights, np.zeros(1)], trainable=False), trainable=False)(errors)  # calculate weighted error by layer
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)
final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  # weight errors by time
model = Model(input=inputs, output=final_errors)
model.compile(loss='mean_absolute_error', optimizer=params['optimizer'])
train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
val_generator = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)
lr_schedule = lambda epoch: params['LR'] if epoch < 75 else params['LR'] #params['LR2']    # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]

if save_model:
    callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))
    
history = model.fit_generator(train_generator, samples_per_epoch, nb_epoch, callbacks=callbacks, validation_data=val_generator, nb_val_samples=N_seq_val / batch_size)

end_time=time_piece.time()
total_training_time = end_time - start_time
with open(HYPO_DIR + "total_model_time.csv", "w") as f:
    f.write(str(total_training_time))

if save_model:
    json_string = model.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
