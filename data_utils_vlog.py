import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, source_file, nt,
                 batch_size=8, vlog_dir="",shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 dim_ordering=K.image_dim_ordering()):
        #self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt
        self.batch_size = batch_size
        self.dim_ordering = dim_ordering
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode
        self.vlog_dir = vlog_dir + "/"
                
        sample_vid_id = self.sources[0].split("-")[0] + "/clip.hkl"
        sample_vid = hkl.load(vlog_dir + "/" + sample_vid_id)
        self.im_shape = sample_vid[0].shape
        #print(self.sources)
        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(len(self.sources) - self.nt) if self.sources[i].split("-")[0] == self.sources[i + self.nt - 1].split("-")[0]])
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < len(self.sources) - self.nt + 1:
                #print(self.sources[curr_location].split("-")[0])
                #print(self.sources[curr_location + self.nt - 1].split("-")[0])
                if self.sources[curr_location].split("-")[0] == self.sources[curr_location + self.nt - 1].split("-")[0]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts
        #print(self.possible_starts)
        self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        #print(self.N_sequences)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.sources[idx:idx+self.nt])
            #batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        file_to_load, frame_number = X[0].split("-")
        frame_number = int(frame_number)
        try:
            first_video = hkl.load(self.vlog_dir + file_to_load + "/clip.hkl")
            num_of_frames_to_use_from_first_vid = first_video.shape[0] - frame_number
            if(num_of_frames_to_use_from_first_vid >= self.nt):
                return first_video[frame_number:frame_number + self.nt].astype(np.float32) / 255
            else:
                clip = np.zeros((self.nt,) + self.im_shape, np.uint8)
                clip[0:num_of_frames_to_use_from_first_vid] = first_video[frame_number:frame_number + num_of_frames_to_use_from_first_vid]
                second_file_to_load, second_frame_number = X[9].split("-")
                second_frame_number = int(second_frame_number)
                second_video = hkl.load(self.vlog_dir + second_file_to_load + "/clip.hkl")
                clip[num_of_frames_to_use_from_first_vid:self.nt] = second_video[:second_frame_number]
            return clip.astype(np.float32) / 255
        except:
            print "data is corrupt:", X
        #num_of_frames_to_use_from_first_vid = first_video.shape[0] - frame_number
        #if(num_of_frames_to_use_from_first_vid >= self.nt):
        #    return first_video[frame_number:frame_number + self.nt].astype(np.float32) / 255
        #else:
        #    clip = np.zeros((self.nt,) + self.im_shape, np.uint8)
        #    clip[0:num_of_frames_to_use_from_first_vid] = first_video[frame_number:frame_number + num_of_frames_to_use_from_first_vid]
        #    second_file_to_load, second_frame_number = X[9].split("-")
        #    second_frame_number = int(second_frame_number)
        #    second_video = hkl.load(self.vlog_dir + second_file_to_load + "/clip.hkl")
        #    clip[num_of_frames_to_use_from_first_vid:self.nt] = second_video[:second_frame_number]
        #return clip.astype(np.float32) / 255

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.sources[idx:idx+self.nt])
        return X_all
