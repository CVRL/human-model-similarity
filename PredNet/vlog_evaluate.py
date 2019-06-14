'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import json
import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils_vlog import SequenceGenerator
from kitti_settings import *
import sys

n_plot = 10
#batch_size = 10
nt = 10
#note we are setting random seed now
np.random.seed(123)

MODEL_DIR = './' + sys.argv[1] + '/'
#DATA_DIR = './vlog_data/'
DATA_DIR = './vlog_generator_data/'
VLOG_DIR = '../../scratch_24/vlog_data/'

RESULTS_SAVE_DIR = MODEL_DIR
weights_file = os.path.join(MODEL_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(MODEL_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

with open(MODEL_DIR + 'hyperparameters.json', 'r') as f:
    params = json.load(f)

batch_size = params['batch_size']

# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
dim_ordering = layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(input=inputs, output=predictions)

#val generator params:
#SequenceGenerator(val_sources, nt, batch_size=batch_size, vlog_dir=VLOG_DIR, N_seq=N_seq_val)
#10000 takes 56 minutes
test_generator = SequenceGenerator(test_sources, nt, batch_size=batch_size, vlog_dir=VLOG_DIR, N_seq=1000, sequence_start_mode='unique', dim_ordering=dim_ordering)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
#predict_generator(self, generator, val_samples, max_q_size=10, nb_worker=1, pickle_safe=False) unbound keras.engine.training.Model method
#X_hat = test_model.predict_generator(test_generator,100)# (len(test_sources) - nt) / 2)

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'vlog_eval_prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()
with open(MODEL_DIR + '/hyperparameters.json', 'r') as f:
    print(f)
    params = json.load(f)
with open(MODEL_DIR + '/vlog_evaulate.json', 'w') as f:
    params['MSE_model'] = float(mse_model)
    params['MSE_prev'] = float(mse_prev)                    
    json.dump(params,f)


# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'vlog_plot_' + str(i) + '.png')
    plt.clf()
