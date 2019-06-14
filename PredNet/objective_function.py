from hyperopt import fmin, tpe, hp, rand, STATUS_OK, STATUS_FAIL
from hyperopt.mongoexp import MongoTrials
import os
import time


def objective(P):
    import json, subprocess
    import os, uuid
    import sys

    #################################
    # dump parameters to './<uuidpath>/hyperparameters.json
    #################################
    exp_key = P[27]
    uuidpath = "./" + exp_key + '/sage_' + str(uuid.uuid4())
    os.makedirs(uuidpath)
    print 'dumping hyperparemters to ./' + uuidpath + '/hyperparameters.json'
    train_data = P[26]
    print(P)
    print 'training on ' + str(train_data)
    metric = P[29]
    print 'returning ' + str(metric) + ' to hyperopt for guidance'
    starttime=time.time()

    with open('./' + uuidpath + '/hyperparameters.json', 'w') as f:
        json.dump({'nb_epoch': P[0],
                   'batch_size': P[1],
                   'samples_per_epoch': P[2],
                   'N_seq_val': P[3],
                   'LR': P[4],
                   'LR2': P[5],
                   'layers': P[6],
                   'num_filters_first_prednet_layer': P[7],
                   'A_filter_1': P[8],
                   'A_filter_2': P[9],
                   'A_filter_3': P[10],
                   'A_filter_4': P[11],
                   'A_filter_5': P[12],
                   'A_filter_6': P[13],
                   'Ahat_filter_1': P[14],
                   'Ahat_filter_2': P[15],
                   'Ahat_filter_3': P[16],
                   'Ahat_filter_4': P[17],
                   'Ahat_filter_5': P[18],
                   'Ahat_filter_6': P[19],
                   'R_filter_1': P[20],
                   'R_filter_2': P[21],
                   'R_filter_3': P[22],
                   'R_filter_4': P[23],
                   'R_filter_5': P[24],
                   'R_filter_6': P[25],
                   'train_data' : P[26],
                   'exp_key': exp_key, #P[27]
                   'optimizer': P[28],
                   'guiding_metric': P[29],
                   'status': 'running',
                   'ID': uuidpath,
                   }, f)

    #################################
    # subprocess call to './train.sh'
    #################################
    if(train_data == 'kitti'):
        print 'running ./train_prednet_on_kitti.sh "' + uuidpath + '"'
        subprocess.call('./train_prednet_on_kitti.sh "' + uuidpath + '"', shell=True)
    else:
        print 'running ./train_prednet_on_vlog.sh "' + uuidpath + '"'
        subprocess.call('./train_prednet_on_vlog.sh "' + uuidpath + '"', shell=True)

    # if training fails, graceful exit
    json_file = os.path.join(uuidpath, 'prednet_kitti_model.json')
    if not (os.path.exists(json_file)):
        return {'status':STATUS_FAIL}
        

    # #################################
    # # create the features and calculate comparison between human and model
    # #################################
    print 'running ./extract_features_from_model_and_compute_similarity_to_human_fMRI.sh "' + uuidpath + '"'
    subprocess.call('./extract_features_from_model_and_compute_similarity_to_human_fMRI.sh "' + uuidpath + '"',
                    shell=True)

    # #################################
    # # calculate MSE on kitti
    # #################################
    print 'running ./eval_on_kitti.sh "' + uuidpath + '"'
    subprocess.call('./eval_on_kitti.sh "' + uuidpath + '"', shell=True)
    print 'running ./eval_on_vlog.sh "' + uuidpath + '"'
    subprocess.call('./eval_on_vlog.sh "' + uuidpath + '"', shell=True)

    # #################################
    # # calculate objectRecognition score (microns phase 1 exp 3)
    # #################################
    print 'running ./test-exp3.sh "' + uuidpath + '"'
    subprocess.call('./test-exp3.sh "' + uuidpath + '"', shell=True)

    # #################################
    # # load from score file
    # #################################
    print 'compiling from hyperparemeters and experiment scores to ./' + uuidpath + '/model_results.json'
    with open('./' + uuidpath + '/score_activation_times_1_to_5.json', 'r') as f:
        scores = json.load(f)
        score = scores['score']
    with open('./' + uuidpath + '/kitti_evaulate.json', 'r') as f:
        params = json.load(f)
        mse_kitti = params['MSE_model']
        mse_previous_kitti = params['MSE_prev']
    with open('./' + uuidpath + '/vlog_evaulate.json', 'r') as f:
        params = json.load(f)
        mse_vlog = params['MSE_model']
        mse_previous_vlog = params['MSE_prev']
    with open('./' + uuidpath + '/object_recoginition_exp3.json', 'r') as f:
        params = json.load(f)
        acc = params['accuracy']
    for key in scores.keys():
        params[key] = scores[key]
    with open('./' + uuidpath + '/model_results.json', 'w') as f:
        params['accuracy'] = acc
        params['simscore'] = score
        params['MSE_kitti'] = mse_kitti
        params['MSE_previous_fame_kitti'] = mse_previous_kitti
        params['MSE_vlog'] = mse_vlog
        params['MSE_previous_fame_vlog'] = mse_previous_vlog
        params['exp_key'] = exp_key
        params['ID'] = uuidpath
        json.dump(params, f)
    #note that we are returning a score to MINIMIZE, so subtract rdm and acc from 1

    if params['guiding_metric'] == 'rdm_sim':
        score = 1 - score
    elif params['guiding_metric'] == 'obj_rec_acc':
        score = 1 - acc
    elif params['guiding_metric'] == 'vlog_mse':
        score = mse_vlog
    elif params['guiding_metric'] == 'kitti_mse':
        score = mse_kitti
    else:
        return {'status':STATUS_FAIL}
    return {
        'loss': score,
        'status': STATUS_OK,
        # -- store other results like this
        'eval_time': time.time(),
        'start_time': starttime,
        'other_stuff': {'acc':acc , 'rdm':score, 'vlog':mse_vlog, 'kitti':mse_kitti},
        }
