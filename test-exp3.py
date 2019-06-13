import json, random, sys
from sklearn.metrics.pairwise import cosine_similarity

# from iarpa_settings import *

if __name__ == '__main__':
    WEIGHTS_DIR = './' + sys.argv[1] + '/'  # overrides the iarpa_settings.WEIGHTS_DIR
    DATA_DIR = './data_exp3/'
    '''
    Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
    '''
    import os
    import numpy as np
    np.random.seed(123) # DO NOT DO THIS WITH THIS SCRIPT
    random.seed(123)
    import hickle as hkl

    from keras.models import Model
    from keras.layers import Input
    from prednet import PredNet

    with open(WEIGHTS_DIR + 'hyperparameters.json', 'r') as f:
        params = json.load(f)

    batch_size = params['batch_size']

    nt = 10

    from keras.models import model_from_json

    with open(WEIGHTS_DIR + 'prednet_kitti_model.json') as f:
        train_model = model_from_json(f.read(), custom_objects={'PredNet': PredNet})
    if len(sys.argv) > 2:
        print "loading weights from " + sys.argv[2]
        weights_file = os.path.join(sys.argv[2])
        train_model.load_weights(os.path.join(weights_file))
    else:   
        train_model.load_weights(os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5'))

    # Create testing model (to output representation layers)
    layer_config = train_model.layers[1].get_config()

    layer_config['output_mode'] = 'R5'
    if (params['layers'] == 5):
        layer_config['output_mode'] = 'R4'
    if (params['layers'] == 4):
        layer_config['output_mode'] = 'R3'
    if (params['layers'] == 3):
        layer_config['output_mode'] = 'R2'
    if (params['layers'] == 2):
        layer_config['output_mode'] = 'R1'
    print("obtaining activations at layer " + layer_config['output_mode'] + ' for object matching')
    test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
    input_shape = list(train_model.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    repLayers = test_prednet(inputs)
    test_model = Model(input=inputs,output=repLayers)

    X_test = hkl.load(os.path.join(DATA_DIR, 'X_val_exp3_kitti_dim.hkl'))
    print(X_test.shape)   
 
    # Reshape for the ariadne task
    selection = np.random.choice(np.arange(0, X_test.shape[0] / 3), X_test.shape[0] / 3 / nt, replace=False)

    #f = open(WEIGHTS_DIR + 'exp3-selections.hkl', 'w')
    #hkl.dump(selection, f)
    
    X_part = [0] * X_test.shape[0]
    index = 0
    for sel in selection:
        # print sel, index
        for i in range(nt):
            X_part[index + i] = X_test[sel * 3].copy()
            for j in range(nt):
                X_part[index + i][j] = X_test[sel * 3, i].copy()
        index += nt
        for i in range(nt):
            X_part[index + i] = X_test[sel * 3 + 1]
            for j in range(nt):
                X_part[index + i][j] = X_test[sel * 3 + 1, i].copy()
        index += nt
        for i in range(nt):
            X_part[index + i] = X_test[sel * 3 + 2]
            for j in range(nt):
                X_part[index + i][j] = X_test[sel * 3 + 2, i].copy()
        index += nt
    del X_test
    X_test = np.array(X_part)
    del X_part

    class_labels = np.array([i / (nt * 3) for i in range(X_test.shape[0])])
    X_test = np.rollaxis(X_test, 3, 2)
    X_test = np.rollaxis(X_test, 4, 3)
    #f = open(os.path.join(DATA_DIR,"test_data_used_for_human_model_sim_evaluation"),"w")
    #hkl.dump(X_test,f)
    ##Uncomment to save example pics
    #import matplotlib
    #matplotlib.use('agg')
    #import matplotlib.pyplot as plt
    #import os
    #if not os.path.isdir('sanity'):
    #    os.mkdir('sanity')
    #for i, image in enumerate(X_test):
    #    plt.imshow(image[0])
    #    plt.savefig("sanity/image"+str(i)+".png")
    X_hat = test_model.predict(X_test, batch_size)


    def create_microns_eval_batch(imgs, ntrials):  # array of images, number of trials
        probes = []
        galleries = []
        grounds = []
        X_test_as_ints = np.arange(X_test.shape[0])
        single_class_mask = np.bincount(class_labels) > 1
        avail_probes = np.arange(single_class_mask.shape[0])[single_class_mask]
        for i in range(ntrials):
            probe_class = np.random.choice(avail_probes, replace=False)
            probe_index = np.random.choice(X_test_as_ints[class_labels == probe_class], replace=False)
            pgallery_index = np.random.choice(X_test_as_ints[class_labels == probe_class], replace=False)

            mask = (class_labels != probe_class)

            gallery = random.sample(X_test_as_ints[mask], 50)
            pos = random.randint(0, len(gallery) - 1)
            gallery[pos] = pgallery_index

            probes.append(probe_index)
            galleries.append(gallery)
            grounds.append(pos)
        return probes, galleries, grounds


    def accuracy(dec, probes, galleries, grounds):
        selections = np.zeros((len(probes),))
        for i in range(len(probes)):
            selections[i] = dec(probes[i], galleries[i], grounds[i])
        correct = selections >= 0
        lgalleries = np.array(map(len, galleries))
        tgallery = np.sum(lgalleries)
        wcorrect = correct * lgalleries
        return float(np.sum(wcorrect)) / tgallery


    def decision(probe, gallery, ground):
        def tmp(id_):
            return X_hat[id_]

        def comp(id1, id2):
            set1 = tmp(id1)
            set2 = tmp(id2)

            dists = np.zeros(nt)
            for i in range(len(set1)):
                dists[i] = cosine_similarity(set1[i].reshape(1, -1), set2[i].reshape(1, -1))[0, 0]

            #import matplotlib
	    #matplotlib.use('agg')
	    #import matplotlib.pyplot as plt
            #import os
            #eval_number = np.where(probe==id1)
            #save_image_path="sanity/" + str(eval_number) + "/"
            #print(X_test.shape)
	    #if not os.path.isdir(save_image_path):
            #    os.mkdir(save_image_path)
	    #plt.imshow(X_test[probe][0])
	    #plt.savefig(save_image_path + "probe_"+str(probe)+".png")
	    #for i, image in enumerate(gallery):
	    #    plt.imshow(X_test[image][0])
	    #    plt.savefig(save_image_path + "gallery_" + str(i) + ".png")
	    return np.mean(dists)

        gdists = map(lambda e: comp(probe, e), gallery)

        pmax = np.argmax(gdists)

        return gdists[pmax] if pmax == ground else -gdists[pmax]


    img_count = X_hat.shape[0]
    probes, galleries, grounds = create_microns_eval_batch(xrange(img_count), 500)
    probe_file = open(DATA_DIR + 'probes_for_accuracy_eval.hkl', 'w')
    gallery_file = open(DATA_DIR + 'galleries_for_accuracy_eval.hkl', 'w')
    ground_truth_file = open(DATA_DIR + 'ground_truth_for_accuracy_eval.hkl', 'w')
    hkl.dump(probes, probe_file)
    hkl.dump(galleries, gallery_file)
    hkl.dump(grounds, ground_truth_file)
    
    acc = accuracy(decision, probes, galleries, grounds)

    with open(WEIGHTS_DIR + 'object_recoginition_exp3.json', 'w') as f:
        params['accuracy'] = acc
        params['status'] = 'success'
        params['rerun'] = 'true'
        json.dump(params, f)
