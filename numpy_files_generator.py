import os
import numpy as np



DATA_PATH = './data'
aff = []
opr = []
vas = []
sen = []
seq_labels = []
video_paths = []

for subject in sorted(os.listdir('{}/Images'.format(DATA_PATH))):
    if (subject != ".DS_Store"):
        subject_path = os.path.join('{}/Images'.format(DATA_PATH), subject)
        if os.path.isdir(subject_path):
            sequence_count = len(os.listdir(subject_path))
            for sequence in sorted(os.listdir(subject_path)):
                seq_path = os.path.join(subject_path, sequence)
                if os.path.isdir(seq_path):
                    seq_data_list = []
                    seq_aff = int(np.loadtxt('{}/Sequence_Labels/AFF/{}/{}.txt'.format(DATA_PATH, subject[2:], sequence)))
                    seq_opr = int(np.loadtxt('{}/Sequence_Labels/OPR/{}/{}.txt'.format(DATA_PATH, subject[2:], sequence)))
                    seq_sen = int(np.loadtxt('{}/Sequence_Labels/SEN/{}/{}.txt'.format(DATA_PATH, subject[2:], sequence)))
                    seq_vas = int(np.loadtxt('{}/Sequence_Labels/VAS/{}/{}.txt'.format(DATA_PATH, subject[2:], sequence)))
    
                    
                    aff.append(seq_aff)
                    opr.append(seq_opr)
                    sen.append(seq_sen)
                    vas.append(seq_vas)

                    video_paths += [subject+"/"+sequence]
seq_labels += [aff]
seq_labels += [opr]
seq_labels += [vas]
seq_labels += [sen]

np.save("./numpy_files/seq_labels",seq_labels)
np.save("./numpy_files/norm_video_paths",video_paths)




#print(video_paths)
#print(seq_labels)

