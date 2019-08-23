import matplotlib.pyplot as plt
from statistics import mean 
import matplotlib.ticker as ticker
import numpy as np
from imports import *


  
def generate_results_dir(result_path,folds,MODEL_PATH_DIR):
  import os
  if not os.path.exists(MODEL_PATH_DIR):
    os.mkdir(MODEL_PATH_DIR)

  if not os.path.exists(result_path):
    os.mkdir(result_path)

  if not os.path.exists(result_path+"/Test"):
    os.mkdir(result_path+"/Test")

  for i in range(folds):
    if not os.path.exists(result_path+"/Fold"+str(i+1)):
      os.mkdir(result_path+"/Fold"+str(i+1))

    for j in range(folds-1):
      if not os.path.exists(result_path+"/Fold"+str(i+1)+"/Fold"+str(j+1)):
        os.mkdir(result_path+"/Fold"+str(i+1)+"/Fold"+str(j+1))


def plot_func(history,num_epochs,path):
  global global_dict

  train_loss = history[:,'train_loss']
  val_loss = history[:,'valid_loss']
  X = [i+1 for i in range(num_epochs)]
  plt.plot(X,train_loss,label="Train")
  plt.plot(X,val_loss, label="Validate")
  #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.legend(loc=2)
  plt.savefig(path+"/Average.png")
  plt.close()

  separate_loss_train = history[:,'separate_loss_train'] 
  separate_loss_train = list(map(lambda x: x.tolist(), separate_loss_train)) 
  separate_loss_val = history[:,'separate_loss_val']
  separate_loss_val = list(map(lambda x: x.tolist(), separate_loss_val)) 

  number = global_dict['labels_dict']['number']
  label = global_dict['labels_dict']['label']
  #print(separate_loss_train)
  for j in range(number):
    train = [i[j] for i in separate_loss_train]
    valid = [i[j] for i in separate_loss_val]
    plt.plot(X,train,label=label[j]+" Train")
    plt.plot(X,valid,label=label[j]+" Validate")
    plt.legend(loc=2)
    plt.savefig(path+"/"+label[j]+".png")
    plt.close()



def write_history(file,net,num_epochs):
  history = net.history

  file.write("epoch                      separate_loss_train                                         train_loss                   separate_loss_val                                             val_loss\n") 
  file.write("-----  ----------------------------------------------------------------------------    ---------  ----------------------------------------------------------------------------    ---------\n")
  for i in range(num_epochs):
      #file.write(str(i)+"     "+str((history[i:i+1,'separate_loss_train'][0]).tolist())+"   "+str(round(history[i:i+1,'train_loss'][0],4))+" "+str((history[i:i+1,'separate_loss_val'][0]).tolist())+"   "+str(round(history[i:i+1,'valid_loss'][0],4))+" "+str(round(history[i:i+1,'dur'][0],2)))
      file.write(str(i+1)+"     "+str((history[i:i+1,'separate_loss_train'][0]).tolist())+"   "+str(round(history[i:i+1,'train_loss'][0],4))+" "+str((history[i:i+1,'separate_loss_val'][0]).tolist())+"   "+str(round(history[i:i+1,'valid_loss'][0],4)))
      file.write("\n")
        

def plot_test(folds,test_fold,test_loss_tensor,path,min_param_idxArray,pred_true_array,typ):
  global global_dict
  X = np.arange(folds+1)
  X=X[1:]

  plt.bar(X, test_fold,align='center', alpha=0.5)
  plt.title("Average Error for data over 5 folds")
  plt.ylabel('Error')
  plt.xlabel('Test Fold #')
  plt.savefig(path+"/"+typ+"_Average_Test.png")
  plt.close()

  number = global_dict['labels_dict']['number']
  label = global_dict['labels_dict']['label']

  fold = folds
  x = 1
  flatten = lambda l: [item for sublist in l for item in sublist]
  y_max = max(flatten(test_loss_tensor))+0.2

  

  bar_width = 0.15
  fig, ax = plt.subplots()
  ticks = [str(i+1) for i in range(folds)]
  plt.xticks(X+1.5*bar_width , ticks)
  color = ['gray',"#8c564b",'#000000',"#8c564b"]

  label = global_dict['labels_dict']['label']
  for i in range(number):
    test = [j[i] for j in test_loss_tensor]
    rects = plt.bar(X+i*bar_width,test,bar_width,color=color[i],alpha=0.6,label=label[i])
  
  
  plt.xlabel('Test Fold #')
  plt.ylabel('Error')
  plt.title('Error Per Label in Actual Intensity Range')
  plt.legend()
  plt.ylim(ymin=0, ymax = y_max)
  plt.tight_layout()
  plt.savefig(path+"/"+typ+"_AllLabelsTest_Together.png")
  plt.close()

  
  file= open(path+"/Test/"+typ+"_Test.txt","w+")
  file.write("Test Error per fold "+str(test_fold)+"\n")
  file.write("Test Error per fold VAS scaled"+str(test_loss_tensor)+"\n")
  file.write("Parameter idx per fold "+str(min_param_idxArray)+"\n")
  file.write("Average loss "+str(mean(test_fold))+"\n")
  print(test_loss_tensor)
  test_loss_tensor = [sum(i) for i in zip(*test_loss_tensor)]
  test_loss_tensor = [i*1.0/folds for i in test_loss_tensor]
  file.write("Average loss per label: "+str(test_loss_tensor)+"\n")
  file.write(str(pred_true_array)+"\n")
  file.close()




