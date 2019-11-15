import matplotlib.pyplot as plt
from statistics import mean 
import matplotlib.ticker as ticker
import numpy as np
from imports import *
import xlsxwriter

  
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

    for j in range(folds):
      if not os.path.exists(result_path+"/Fold"+str(i+1)+"/Fold"+str(j+1)):
        os.mkdir(result_path+"/Fold"+str(i+1)+"/Fold"+str(j+1))

#change here when changung global dict
def plot_func(history,num_epochs,path):
  global global_dict
  #train_val = history[:,'train_val_bool_']
  train_val = global_dict["train_val"] 

  train_loss = history[:,'train_loss']
  val_loss = history[:,'valid_loss']
  X = [i+1 for i in range(num_epochs)]
  plt.plot(X,train_loss,label="Train")
  plt.title(typ+"Loss over the Epochs")
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  if not train_val:
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
    if not train_val:
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
        



def pred_true_bar(path,folds,pred_true_array):
  number = global_dict['labels_dict']['number']
  label = global_dict['labels_dict']['label']

  for f in range(folds):
    pred_true = pred_true_array[f]
    pred = pred_true[0]
    true = pred_true[1]
    
    for n in range (number):
      pred = [i[n] if i[n]!=0 else 0+0.05 for i in pred]
      true = [i[n] if i[n]!=0 else 0+0.05 for i in true]
      X = np.arange(len(pred))
      #Calculate optimal width
      width = np.min(np.diff(X))/3
      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.bar(X-width/2,pred,width,color='orange',label='Predicted')
      ax.bar(X+width/2,true,width,color='blue',label='True')
      ax.set_ylim(-1,10)
      ax.set_xlabel('Video Sequences in Test')
      ax.set_ylabel('Pain Intensity')
      plt.legend(loc='upper right')
      plt.title('Predicted vs True labels')

      plt.grid()
      plt.savefig(path+"/Test/"+"predTrue_distribution_"+str(f+1)+str(label[n])+".png")
      plt.close()


def excel_file(path,pred_true_array):
  workbook = xlsxwriter.Workbook(path+'/arrays.xlsx')
  worksheet = workbook.add_worksheet()

  row = 3
  column = 0

  for col, data in enumerate(pred_true_array):
    pred = list(map(lambda x: x[0].tolist(), data[0]))
    true = list(map(lambda x: x[0].tolist(), data[1]))
    worksheet.write_column(row, column, pred)
    worksheet.write_column(row, column+1, true)
    column +=4

  workbook.close()


def plot_test(folds,test_fold,test_loss_tensor,path,min_param_idxArray,pred_true_array,labels_error_array,labels_error_array_perfold,typ):
  global global_dict
  w = list(map(lambda x: int(x),global_dict["labels_dict"]["w"].tolist()))
  X = np.arange(folds+1)
  X=X[1:]

  plt.bar(X, test_fold,align='center', alpha=0.5)
  plt.title(typ+" Average Error for data over 5 folds")
  plt.ylabel('Error')
  plt.xlabel('Test Fold #')
  plt.grid()
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
    rects = plt.bar(X+i*bar_width,test,bar_width,align='center',color=color[i],alpha=0.6,label=label[i])
  
  
  plt.xlabel('Test Fold #')
  plt.ylabel('Error')
  plt.title(typ+' Error Per Fold in Actual Intensity Range')
  plt.legend()
  plt.ylim(ymin=0, ymax = y_max)
  plt.tight_layout()
  plt.grid()
  plt.savefig(path+"/Test/"+typ+"_AllLabelsTest_Together.png")
  plt.close()

  print("labels_error_array",labels_error_array)
  for f in range (folds):
    for i in range(number):
      X = np.arange(len(labels_error_array_perfold[f][i]))
      X_str = [str(i) for i in X]
      Y = labels_error_array_perfold[f][i]
      #new line
      #mean_err = test_loss_tensor[f][i]
      plt.xticks(X, X_str)
      plt.bar(X,Y,align='center',color=color[i],alpha=0.6,label=label[i])
      #new line
      #plt.gcf().text(0,0,"Mean: "+ str(round(mean_err,4)), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
      #plt.subplots_adjust(left=0.25)

      plt.xlabel('Error per Intensity#')
      plt.ylabel('Error')
      plt.title(typ+' Error per Intensity in Actual Intensity Range')
      plt.legend()
      #plt.xlim(xmin=0, xmax = w[i]+1)
      plt.tight_layout()
      plt.grid()
      plt.savefig(path+"/Test/"+typ+"_ErrorPerIntensity_"+str(f+1)+"_"+str(label[i])+".png")
      plt.close()

  #plotting per label error avearge 
  for i in range(number):
    X = np.arange(len(labels_error_array[i]))
    X_str = [str(i) for i in X]
    Y = labels_error_array[i]
    plt.xticks(X, X_str)
    plt.bar(X,Y,align='center',color=color[i],alpha=0.6,label=label[i])

    plt.xlabel('Error per Intensity#')
    plt.ylabel('Error')
    plt.title(typ+' Average Error per Intensity in Actual Intensity Range')
    plt.legend()
    #plt.xlim(xmin=0, xmax = w[i]+1)
    plt.tight_layout()
    plt.grid()
    plt.savefig(path+"/Test/"+typ+"_ErrorPerIntensity_"+str(label[i])+".png")
    plt.close()

  print('pred true',pred_true_array)
  for i in range (folds):
    pred_true = pred_true_array[i]
    pred = pred_true[0]
    true = pred_true[1]
    X = np.arange(len(pred))

    for n in range (number):
      Y_pred = [pred[p][n] for p in range (len(pred))]
      Y_true = [true[p][n] for p in range (len(true))]
      print("y pred", Y_pred)
      plt.ylim(0, w[n])
      plt.plot(X,Y_pred,color='orange',label='Predicted')
      plt.plot(X,Y_true, color='blue',label='True')
      plt.title("Predicted VS True Values for "+str(label[n])+" fold#: "+str(i+1))
      plt.legend(loc='upper right')
      plt.ylabel('Pain Intensity')
      plt.xlabel('Data Points')
      plt.grid()
      plt.savefig(path+"/Test/"+"predTrue_fold"+str(i+1)+str(label[n])+".png")
      plt.close()

  pred_true_array_zipped = []
  for i in range (folds):
    pred_true = pred_true_array[i]
    pred = pred_true[0]
    true = pred_true[1]
    zipped = [(pred[p],true[p]) for p in range (len(pred))]
    pred_true_array_zipped += zipped

  print('zipped',pred_true_array_zipped)


  file= open(path+"/Test/"+typ+"_Test.txt","w+")
  file.write("Test Error per fold "+str(test_fold)+"\n")
  file.write("Test Error per fold VAS scaled"+str(test_loss_tensor)+"\n")
  file.write("Parameter idx per fold "+str(min_param_idxArray)+"\n")
  file.write("Average loss "+str(mean(test_fold))+"\n")
  print(test_loss_tensor)
  test_loss_tensor = [sum(i) for i in zip(*test_loss_tensor)]
  test_loss_tensor = [i*1.0/folds for i in test_loss_tensor]
  file.write("Average loss per label: "+str(test_loss_tensor)+"\n")

  file.write("Average loss per label per intensity: "+str(labels_error_array)+"\n")


  file.write("Average loss per intrensity per fold"+"\n")
  for f in range (folds):
    file.write("FOLD: "+str(f)+"\n")
    for i in range(number):
      file.write(str(labels_error_array_perfold[f][i])+"\n")
    file.write("\n")

  length = int(len(labels_error_array)/folds)
  file.write("Predict Array"+"\n")
  for i in range (folds-1):
    file.write(str(pred_true_array_zipped[i*length:(i+1)*length])+"\n")
    file.write("\n")
  
  file.write(str(pred_true_array_zipped[(folds-1)*length:])+"\n")

  file.close()

  pred_true_bar(path,folds,pred_true_array)
  excel_file(path,pred_true_array)





