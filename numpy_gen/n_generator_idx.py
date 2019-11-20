import os
import numpy as np
import argparse
import random



if __name__ == '__main__':
    parser = argparse.ArgumentParser( prog='Numpy Generator',prefix_chars='-')
    parser.add_argument("-Input", default="../UNBC_Warped_Videos", help="put warped input images path")
    parser.add_argument("-Sequence", default="../UNBC_Warped_Videos", help="put original sequence laebls path")
    parser.add_argument("-Output", default='./numpy_files/', help="put output images path")
    parser.add_argument("-Random", type=bool, default=True, help="True if want to randomly shuffle data")

    args = parser.parse_args()

    path = args.Input
    pathW = args.Output
    path_seq = args.Sequence
    random_flag = args.Random

    if (random_flag):
        all_folds = ["064-ak064","121-vw121","047-jl047","096-bg096","059-fn059",\
        "107-hs107","120-kz120","097-gf097","106-nm106","092-ch092",\
        "095-tv095","043-jh043","103-jk103","048-aa048","042-ll042",\
        "109-ib109","108-th108",'123-jh123','124-dn124','066-mg066',\
        "049-bm049","115-jy115","052-dr052","101-mg101","080-bn080"]
        
        all_folds.sort()
        random.shuffle(all_folds)
        all_folds =[all_folds[5*x:5*(x+1)] for x in range(5)]
        fold1 = all_folds[0]
        fold2 = all_folds[1]
        fold3 = all_folds[2]
        fold4 = all_folds[3]
        fold5 = all_folds[4]

    else:
        fold1 = ["064-ak064","121-vw121","047-jl047","096-bg096","059-fn059"]
        fold2 = ["107-hs107","120-kz120","097-gf097","106-nm106","092-ch092"]
        fold3 = ["095-tv095","043-jh043","103-jk103","048-aa048","042-ll042"]
        fold4 = ["109-ib109","108-th108",'123-jh123','124-dn124','066-mg066']
        fold5 = ["049-bm049","115-jy115","052-dr052","101-mg101","080-bn080"]


    # if mini
    #fold1 = ["042-ll042"]
    #fold2 = ["043-jh043"]
    #fold3 =["047-jl047"]
    #fold4 = ["048-aa048"]
    #fold5= ["049-bm049"]

    DATA_PATH = path
    aff = []
    opr = []
    vas = []
    sen = []
    video_paths = []

    fold1_idx =[]
    fold2_idx =[]
    fold3_idx =[]
    fold4_idx =[]
    fold5_idx =[]

    

    seq_labels = []

    seq_idx=0

    for subject in sorted(os.listdir('{}/Images'.format(DATA_PATH))):
        # if subject in fold1:
        #     print(subject," fold1")
        # if subject in fold2:
        #     print(subject," fold2")
        # if subject in fold3:
        #     print(subject," fold3")
        # if subject in fold4:
        #     print(subject," fold4")
        # if subject in fold5:
        #     print(subject," fold5")

        if (subject != ".DS_Store"):
            subject_path = os.path.join('{}/Images'.format(DATA_PATH), subject)
            if os.path.isdir(subject_path):
                sequence_count = len(os.listdir(subject_path))
                for sequence in sorted(os.listdir(subject_path)):
                    seq_path = os.path.join(subject_path, sequence)
                    if os.path.isdir(seq_path):
                        seq_data_list = []
                        seq_aff = int(np.loadtxt('{}/Sequence_Labels/AFF/{}/{}.txt'.format(path_seq, subject[:], sequence)))
                        seq_opr = int(np.loadtxt('{}/Sequence_Labels/OPR/{}/{}.txt'.format(path_seq, subject[:], sequence)))
                        seq_sen = int(np.loadtxt('{}/Sequence_Labels/SEN/{}/{}.txt'.format(path_seq, subject[:], sequence)))
                        seq_vas = int(np.loadtxt('{}/Sequence_Labels/VAS/{}/{}.txt'.format(path_seq, subject[:], sequence)))

                        aff.append(seq_aff)
                        opr.append(seq_opr)
                        sen.append(seq_sen)
                        vas.append(seq_vas)
                        video_paths += [subject+"/"+sequence]

                        if subject in fold1:
                            fold1_idx+=[seq_idx]

                        elif subject in fold2:
                            fold2_idx+=[seq_idx]

                        elif subject in fold3:
                            fold3_idx+=[seq_idx]


                        elif subject.strip() in fold4:
                            fold4_idx+=[seq_idx]


                        elif subject in fold5:
                            print(seq_opr)
                            fold5_idx+=[seq_idx]

                        else:
                            print(subject,'in no folds')
                        seq_idx +=1





    if not os.path.exists(pathW):
        os.makedirs(pathW)

   
    seq_labels = []
    seq_labels += [aff]
    seq_labels += [opr]
    seq_labels += [vas]
    seq_labels += [sen]

    seq_idx_arr = []
    seq_idx_arr +=[fold1_idx]
    seq_idx_arr +=[fold2_idx]
    seq_idx_arr += [fold3_idx]
    seq_idx_arr += [fold4_idx]
    seq_idx_arr += [fold5_idx]
    print(seq_idx_arr)
    print(all_folds)
    print(aff)

    #np.save(pathW+"/seq_labels",seq_labels)
    #np.save(pathW+"/norm_video_paths",video_paths)
    flatten = lambda l: [item for sublist in l for item in sublist]
    print(len(flatten(seq_idx_arr)))
    np.save(pathW+"/seq_idx_mini",seq_idx_arr)


    


