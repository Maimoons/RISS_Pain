import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2

#Importing other files
from model import CnnRnn

#can change this accordingly
#from dataloader_image import PainDataset

from dataloader_video import PainDataset

import numpy as np
import time
import os
import pickle

from skorch.dataset import CVSplit
from skorch import NeuralNetRegressor
from skorch.helper import SliceDataset
from skorch.helper import predefined_split
from skorch.utils import is_dataset
from skorch.callbacks import Callback,LRScheduler,Checkpoint,ProgressBar,BatchScoring,EpochScoring

from sklearn.model_selection import cross_val_predict, GridSearchCV

from collections import OrderedDict

#Global array for per epoch loss on the same scale as the label
global_dict = {}
def initialise_globalDict(Test):
  print('initialising dict')
  global global_dict
  folds = 5
  timestamp = time.strftime("%d%m-%H%M")
  MODEL_PATH_DIR = './models_OPR_Random3'
  result_path = "./"+timestamp+"Testdel__FoldResult_OPR_TwoLevel_Random3"  

  if Test:
    #TEST_MODEL_PATH_DIR = './models_test'
    TEST_MODEL_PATH_DIR = './2test_models_OPR_Ordered'
    test_result_path = "./"+timestamp+"_justtestingTest" 
    if not os.path.exists(test_result_path): 
      os.mkdir(test_result_path)
      os.mkdir(test_result_path+'/Test')

    global_dict['TEST_MODEL_PATH_DIR'] = TEST_MODEL_PATH_DIR
    global_dict['test_result_path'] = test_result_path
    #global_dict['TEST_DATA_PATH'] = "../data/test_images/"
    global_dict['TEST_DATA_PATH'] = "../../UNBC_Warped_Videos/Images/"
    files= {
    #"seq_labels" : '../test_numpy_files/seq_labels.npy',
    #"video_paths" : '../test_numpy_files/norm_video_paths.npy',
    "seq_labels" : '../numpy_files_ordered/seq_labels.npy',
    "video_paths" : '../numpy_files_ordered/norm_video_paths.npy',
  }
  

  else:
    generate_results_dir(result_path,folds,MODEL_PATH_DIR)
    files= {
    "seq_labels" : '../numpy_files_ordered/seq_labels.npy',
    "video_paths" : '../numpy_files_ordered/norm_video_paths.npy',
    #"seq_labels" : '../numpy_files_random/seq_labels.npy',
    #"video_paths" : '../numpy_files_random/norm_video_paths.npy',
  }
  #global_dict["all_idx_array"] = np.load('../numpy_files_ordered/seq_idx.npy').tolist()
  global_dict["all_idx_array"] = np.load('../numpy_files_random/seq_idx3.npy').tolist()
  
  DATA_PATH = "../../UNBC_Warped_Videos/Images/"
  #DATA_PATH = "../../UNBC_Warped_Images/Images/"
  

  train_params = {
      #"cuda": False,
      #"device": 'cpu',
      "device": 'cuda',
      "cuda": True,
      "seed": 1,
      "batch_size": 1,
      "test_batch_size": 1,
      "epochs": 15,
      "lr": [0.0001,0.0001,0.0001,0.0001],
      "weight_decay": 0,
      "console_logs": 200,
      "training_loss_func": 'mse',
      "regularization": True,
      "custom_loss_alpha": 0.7,
  }
  w = torch.FloatTensor([5])
  if (train_params["cuda"]):
    w = w.cuda()
  
  labels_dict = {
    "w" : w,
    #idx is the index range of the labels
    "idx":[1,2],
    "number":1,
    #"label":['AFF,OPR,VAS,SEN']
    "label":['OPR']
  }

  network_params = {
      "pre_trained": True,
      "input_size": 4096,
      "hidden_size": 1024,
      "num_layers": [2,3],
      "nonlinearity": 'tanh',
      "bias": True,
      "batch_first": True,
      "dropout": [0.2,0.3],
      "bidirectional": False,
      "lambda": [0.000001,0.000005],
  }
  global_dict['DATA_PATH'] = DATA_PATH
  global_dict["train_params"] = train_params
  global_dict["network_params"] = network_params
  global_dict["loss_tensor_train"] = torch.zeros([1,labels_dict['number']],device=torch.device(global_dict["train_params"]["device"]))
  global_dict["loss_tensor_val"] = torch.zeros([1,labels_dict['number']],device=torch.device(global_dict["train_params"]["device"]))
  global_dict["num_seq_train"] = 0
  global_dict["num_seq_val"] = 0
  global_dict["num_seq_train_val"] = 0
  global_dict["train_val"] = False
  global_dict["num_epochs"] = 30
  global_dict["Ifold"] = 0
  global_dict['files'] = files
  global_dict['MODEL_PATH_DIR'] = MODEL_PATH_DIR
  global_dict['result_path'] = result_path
  global_dict['random'] = False
  global_dict['folds'] = folds
  global_dict['labels_dict'] = labels_dict
  global_dict['Test'] = Test



from plot import *



