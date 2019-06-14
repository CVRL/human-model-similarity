import os
import numpy as np
from six.moves import cPickle
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *
import hickle as hkl
import scipy.io as sio
import sys
import json
import rdm_sim_score as rdm

#number of stimuli
n_plot = 92

#batch size for test model
batch_size = 10

#number of timesteps (number of times stimuli will be viewed)
nt = 10

MODEL_DIR = './' + sys.argv[1] + '/'
if len(sys.argv) > 2:
    weights_file = os.path.join(sys.argv[2])
else:
    weights_file = os.path.join(MODEL_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(MODEL_DIR, 'prednet_kitti_model.json')
print weights_file
with open(MODEL_DIR + 'hyperparameters.json', 'r') as f:
    params = json.load(f)


# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)
#not implemented yet :) 
feature_vectors = []

layers = params['layers']
layer_list = []
for i in range(0,layers):
    layer_list.append('R' + str(i))
#    layer_list.append('Ahat' + str(i))
#    layer_list.append('A' + str(i))
#    layer_list.append('E' + str(i))

X_test = hkl.load(os.path.join('stimuli_test_data.hkl'))
X_test = np.rollaxis(X_test, 3, 2)
X_test = np.rollaxis(X_test, 4, 3)

#discard first prediction (black screen)
#discard last few predictions because activations level (although this makes almost no difference)
times_1_to_5 = []
for layer in layer_list:
    # Create testing model (to output predictions)
    layer_config = train_model.layers[1].get_config()
    layer_config['output_mode'] = layer
    dim_ordering = layer_config['dim_ordering']
    test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)
    test_model = Model(input=inputs, output=predictions)
       
    #squash features together
    X_hat = test_model.predict(X_test, batch_size)
    X_hat = X_hat[:,-1,:]
    X_hat_time = X_hat.reshape(92,10,-1)
    #here we discard the extra timesteps
    X_hat_time_1_to_5 = X_hat_time[:,1:5]
    X_hat_time_1_to_5 = X_hat_time_1_to_5.reshape(92,-1)
    times_1_to_5.append(X_hat_time_1_to_5)

#concat all layers' features together into one set of features
time_1_to_5 = np.concatenate(times_1_to_5,axis=1)
individual_rdms_dict = rdm.compare_all_human_rdms_to_model(time_1_to_5)
print("comparing to general human rdm ...")
score = rdm.compare_numpy_activations_from_model_to_human(time_1_to_5)
individual_rdms_dict["score"] = score
print(score)
print("saving")
with open('./' + sys.argv[1] + '/score_activation_times_1_to_5.json', 'w') as f:
    json.dump(individual_rdms_dict, f)
