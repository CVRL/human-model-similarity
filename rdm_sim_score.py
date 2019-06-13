import hickle as hkl
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import scipy.io as sio
import os

fmri_activations_to_stimuli="features_from_human_stimuli.hkl"
fmri_activations_individual_brains="rsatoolbox/Demos/92imageData/92_brainRDMs.mat"

def unwrap_square_form(sq_form_matrix):
    return squareform(sq_form_matrix)

def wrap_square_form(non_square_form_array):
    return squareform(non_square_form_array)

def compute_square_rdm_from_features(activations):
    #subtract out the mean
    #https://stackoverflow.com/questions/8946810/is-there-an-equivalent-to-the-matlab-function-bsxfun-in-python
    mean = np.mean(activations,axis=1)
    #print(mean.shape)
    subbed_out = activations - mean.reshape(92,1)
    #print(subbed_out.shape)
    normalized = normalize(subbed_out, axis=1)
    #print(normalized.shape)
    pdistd = pdist(normalized, metric="correlation")
    #print(pdistd.shape)
    square = squareform(pdistd)
    #print(square.shape)
    return square

def compute_rdm_correlation(model_rdm,human_rdm):
    model = unwrap_square_form(model_rdm)
    human = unwrap_square_form(human_rdm)
    corr, pval = spearmanr(model,human)
    return corr

def compare_saved_activations_from_model_to_human(path_to_weights):
    model_activations = hkl.load(os.path.join(path_to_weights,fmri_activations_to_stimuli))
    return compare_numpy_activations_from_model_to_human(model_activations)

def compare_numpy_activations_from_model_to_human(model_activations):
    model_rdm = compute_square_rdm_from_features(model_activations)
    human_rdm = sio.loadmat("ave_hum_rdm.mat")
    human_rdm = human_rdm['ave_hum_rdm']
    human_rdm = human_rdm[0][0][0]
    return compute_rdm_correlation(model_rdm,human_rdm)

def compare_all_human_rdms_to_model(model_activations):
    human_matlab = sio.loadmat(fmri_activations_individual_brains)
    human_rdms = human_matlab.get('RDMs')
    model_rdm = compute_square_rdm_from_features(model_activations)

    comparison_dict = dict()
    for p1 in range(0, 4):
        for s1 in range(0, 2):
            print("(" + str(p1) + " " + str(s1) + ")")
            first_p_first_s = human_rdms.item(0, p1, s1)
            first_p_first_s_rdm = first_p_first_s[0]
            score = compute_rdm_correlation(first_p_first_s_rdm, model_rdm)
            comparison_dict["(" + str(p1) + " " + str(s1) + ")"] = score
    return comparison_dict

def compare_all_average_by_session_human_rdms_to_model(model_activations):
    model_rdm = compute_square_rdm_from_features(model_activations)
    comparison_dict = dict()
    for session in ['RDMs_hIT_bySessions_1','RDMs_hIT_bySessions_2']:
        human_matlab = sio.loadmat(session + '.mat'))
        human_rdms = human_matlab[session]
        human_rdms = human_rdms.item(0, 0)
        human_rdms = human_rdms[0]
        score = compute_rdm_correlation(human_rdms, model_rdm)
        comparison_dict[session] = score
    return comparison_dict
