from hyperopt import fmin, tpe, hp, rand
from hyperopt.mongoexp import MongoTrials
import os
import time
from objective_function_recreating_experiment_conditions import objective
import sys
exp_key = sys.argv[1]
#exp_key = "randomly_initialize_archs"
if __name__ == '__main__':
    # This is the set of hyperparameters and their search space,
    # see hyperopt documentation for proper param definition
    # space = (hp.uniform('a', -2, 2), hp.uniform('b', -2, 2))
    space = (hp.choice('nb_epoch', [i for i in range(10, 400, 10)]),  # 150
             hp.choice('batch_size', [i for i in range(2, 9, 2)]),  # 4
             hp.choice('samples_per_epoch', [i for i in range(100, 2000, 100)]),  # 500
             hp.choice('N_seq_val', [i for i in range(100, 2000, 100)]),  # 100
             hp.choice('LR', [0.1, 0.01,0.001,0.0001,0.00001]),
             hp.choice('LR2', [0.1,0.01,0.001,0.0001,0.00001]),
             hp.choice('layers', [4]),
             hp.choice('num_filters_first_prednet_layer', [i for i in range(48,49)]),
             hp.choice('A_filter_1', [i for i in range(1, 9)]),
             hp.choice('A_filter_2', [i for i in range(1, 9)]),
             hp.choice('A_filter_3', [i for i in range(1, 9)]),
             hp.choice('A_filter_4', [i for i in range(1, 9)]),
             hp.choice('A_filter_5', [i for i in range(1, 9)]),
             hp.choice('A_filter_6', [i for i in range(1, 9)]),
             hp.choice('Ahat_filter_1', [i for i in range(1, 9)]),
             hp.choice('Ahat_filter_2', [i for i in range(1, 9)]),
             hp.choice('Ahat_filter_3', [i for i in range(1, 9)]),
             hp.choice('Ahat_filter_4', [3]),
             hp.choice('Ahat_filter_5', [i for i in range(1, 9)]),
             hp.choice('Ahat_filter_6', [i for i in range(1, 9)]),
             hp.choice('R_filter_1', [i for i in range(1, 9)]),
             hp.choice('R_filter_2', [i for i in range(1, 9)]),
             hp.choice('R_filter_3', [i for i in range(1, 9)]),
             hp.choice('R_filter_4', [3]),
             hp.choice('R_filter_5', [i for i in range(1, 9)]),
             hp.choice('R_filter_6', [i for i in range(1, 9)]),
             hp.choice('train_data',['kitti']),
             hp.choice('exp_key',[exp_key]),
             hp.choice('optimizer',['Adam',]),
             hp.choice('guiding_metric',['rdm_sim']),
             )


    def suggest(a, b, c):
        seed = int(time.time())
        return rand.suggest(a, b, c, seed=seed)


    # single rn
    best = fmin(objective, space, algo=suggest, max_evals=25)
    print(best)
