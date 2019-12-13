import matplotlib.pyplot as plt
from statistics import mean 
import matplotlib.ticker as ticker
import numpy as np
from imports import *
import xlsxwriter

  
def generate_results_dir(result_path,folds,MODEL_PATH_DIR,gs_combo):
  import os
  if not os.path.exists(MODEL_PATH_DIR):
    os.mkdir(MODEL_PATH_DIR)

  if not os.path.exists(MODEL_PATH_DIR+"/Test"): 
    os.mkdir(MODEL_PATH_DIR+"/Test")

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

      if gs_combo != 1:
        for c in range(gs_combo):
          if not os.path.exists(result_path+"/Fold"+str(i+1)+"/Fold"+str(j+1)+"/param_"+str(c)):
            os.mkdir(result_path+"/Fold"+str(i+1)+"/Fold"+str(j+1)+"/param_"+str(c))




#change here when changung global dict
def plot_func(history,num_epochs,path):
  global global_dict
  
  train_val = global_dict["train_val"] 
  train_loss = history[:,'train_loss']
  val_loss = history[:,'valid_loss']

  X = [i+1 for i in range(num_epochs)]
  plt.plot(X,train_loss,color='blue',label="Train")
  plt.title("Average Loss over the Epochs")
  plt.legend(loc=1)
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.savefig(path+"/AverageLoss_train.png")
  #if not train_val:
  plt.plot(X,val_loss,color='orange', label="Validate")
  plt.legend(loc=1)
  plt.savefig(path+"/AveragLoss_trainval.png")
  plt.close()

  plt.title("Loss over the Epochs")
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.plot(X,val_loss,color='orange', label="Validate")
  plt.legend(loc=1)
  plt.savefig(path+"/AverageLoss_val.png")
  plt.close()



  loss_tensor_train_MAE = history[:,"loss_tensor_train_MAE"] 
  loss_tensor_train_MAE = list(map(lambda x: x.tolist(),loss_tensor_train_MAE)) 
  loss_tensor_train_MSE = history[:,"loss_tensor_train_MSE"] 
  loss_tensor_train_MSE = list(map(lambda x: x.tolist(),loss_tensor_train_MSE)) 
  
  loss_tensor_val_MAE = history[:,"loss_tensor_val_MAE"] 
  loss_tensor_val_MAE = list(map(lambda x: x.tolist(),loss_tensor_val_MAE)) 
  loss_tensor_val_MSE = history[:,"loss_tensor_val_MSE"] 
  loss_tensor_val_MSE = list(map(lambda x: x.tolist(),loss_tensor_val_MSE)) 
  
  number = global_dict['labels_dict']['number']
  label = global_dict['labels_dict']['label']

  plt.title("MAE Scaled Loss over the Epochs")
  plt.ylabel('Scaled Loss')
  plt.xlabel('Epochs')
  for j in range(number):
    train = [i[j] for i in loss_tensor_train_MAE]
    valid = [i[j] for i in loss_tensor_val_MAE]

    plt.plot(X,train,color='blue',label=label[j]+" Train")
    plt.legend(loc=1)
    plt.savefig(path+"/MAELoss_train_"+label[j]+".png")
    #if not train_val:
    plt.plot(X,valid,color='orange',label=label[j]+" Validate")
    plt.legend(loc=1)
    plt.savefig(path+"/MAELoss_trainval_"+label[j]+".png")
    plt.close()

    plt.title("MAE Scaled Loss over the Epochs")
    plt.ylabel('Scaled Loss')
    plt.xlabel('Epochs')
    plt.plot(X,valid,color='orange',label=label[j]+" Validate")
    plt.legend(loc=1)
    plt.savefig(path+"/MAELoss_val_"+label[j]+".png")
    plt.close()

  plt.title("MSE Scaled Loss over the Epochs")
  plt.ylabel('Scaled Loss')
  plt.xlabel('Epochs')
  for j in range(number):
    train = [i[j] for i in loss_tensor_train_MSE]
    valid = [i[j] for i in loss_tensor_val_MSE]

    plt.plot(X,train,color='blue',label=label[j]+" Train")
    plt.legend(loc=1)
    plt.savefig(path+"/MSELoss_train_"+label[j]+".png")
    #if not train_val:
    plt.plot(X,valid,color='orange',label=label[j]+" Validate")
    plt.legend(loc=1)
    plt.savefig(path+"/MSELoss_trainval_"+label[j]+".png")
    plt.close()

    plt.title("MSE Scaled Loss over the Epochs")
    plt.ylabel('Scaled Loss')
    plt.xlabel('Epochs')
    plt.plot(X,valid,color='orange',label=label[j]+" Validate")
    plt.legend(loc=1)
    plt.savefig(path+"/MSELoss_val_"+label[j]+".png")
    plt.close()



def write_history(file,net,num_epochs):
  history = net.history

  file.write("epoch                      loss_tensor_train_MAE                                                         loss_tensor_train_MSE                                      train_loss                   loss_tensor_val_MAE                                                                loss_tensor_val_MSE                                        val_loss\n") 
  file.write("-----  ----------------------------------------------------------------------------  ----------------------------------------------------------------------------    ---------  ----------------------------------------------------------------------------     ----------------------------------------------------------------------------    ---------\n")
  for i in range(num_epochs):
      #file.write(str(i)+"     "+str((history[i:i+1,'separate_loss_train'][0]).tolist())+"   "+str(round(history[i:i+1,'train_loss'][0],4))+" "+str((history[i:i+1,'separate_loss_val'][0]).tolist())+"   "+str(round(history[i:i+1,'valid_loss'][0],4))+" "+str(round(history[i:i+1,'dur'][0],2)))
      file.write(str(i+1)+"     "+str((history[i:i+1,'loss_tensor_train_MAE'][0]).tolist())+"     "+str((history[i:i+1,'loss_tensor_train_MSE'][0]).tolist())+"   "+str(round(history[i:i+1,'train_loss'][0],4))+" "+str((history[i:i+1,'loss_tensor_val_MAE'][0]).tolist())+" "+str((history[i:i+1,'loss_tensor_val_MSE'][0]).tolist())+"   "+str(round(history[i:i+1,'valid_loss'][0],4)))
      file.write("\n")
        



def pred_true_bar(path,f,pred_true_array,w):
  number = global_dict['labels_dict']['number']
  label = global_dict['labels_dict']['label']
  #for f in range(folds):
  pred_true = pred_true_array[f]
  pred = pred_true[0]
  true = pred_true[1]
    
  for n in range (number):
    pred = [i[n] if i[n]!=0 else 0+0.05 for i in pred]
    true = [i[n] if i[n]!=0 else 0+0.05 for i in true]
    X = np.arange(len(pred))
    #Calculate optimal width
    width = np.min(np.diff(X+[1]))/3
    #width = 0.5
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
    plt.savefig(path+"/Test/"+"predTrue_bars_"+str(label[n])+"_Fold:"+str(f+1)+".png")
    plt.close()

  #print('pred true',pred_true_array)
  #for i in range (folds):
  pred_true = pred_true_array[f]
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
    plt.title("Predicted VS True Values for "+str(label[n])+" fold#: "+str(f+1))
    plt.legend(loc='upper right')
    plt.ylabel('Pain Intensity')
    plt.xlabel('Data Points')
    plt.grid()
    plt.savefig(path+"/Test/"+"predTrue_lines_"+str(label[n])+"_Fold:"+str(f+1)+".png")
    plt.close()


def excel_file(worksheet,f,path,pred_true_array):
  global global_dict
  folds = global_dict['folds']
  number = global_dict['labels_dict']['number']
  label = global_dict['labels_dict']['label']

  worksheet.write(0, f*4, 'Test '+str(f+1))
  base = 4*f*number
  for c in range(number):
    worksheet.write(2,base+4*c,'Pred')
    worksheet.write(2,base+4*c +1,'True')
    worksheet.write(2,base+4*c +2,'MSE')
    worksheet.write(2,base+4*c+3,'MAE')


  row = 3
  column = 0

  for col, data in enumerate(pred_true_array):
    pred = list(map(lambda x: x[0].tolist(), data[0]))
    true = list(map(lambda x: x[0].tolist(), data[1]))
    worksheet.write_column(row, column, pred)
    worksheet.write_column(row, column+1, true)
    column += 4*number



def plot_confusion_matrix(f,path,y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred,labels=classes)
    classes = classes

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    plt.autoscale()
    return ax


def confusion(f,path,pred_true_array,pain_labels,w):
  #for data in pred_true_array:
  #-1 to get the last fold
  for i,pain_label in enumerate(pain_labels):
    labels = [l for l in range(w[i]+1)]
    data = pred_true_array[-1]
    actual = list(map(lambda x: x[0].tolist(), data[1]))
    predicted = list(map(lambda x: x[0].tolist(), data[0]))
    plot_confusion_matrix(f,path,actual,predicted, classes=labels,
      title='Confusion matrix')
    plt.savefig(path+"/Test/"+pain_label+"_confusion_Fold:"+str(f+1)+".png")
    plt.close()
    #results = confusion_matrix(actual, predicted,labels=labels)
    

#this path has the combination number as well
def plot_test_onefold(worksheet,f,folds,test_fold,test_loss_tensor,path,\
  min_param_idxArray,pred_true_array,labels_error_array_perfold,typ):
  global global_dict
  number = global_dict['labels_dict']['number']
  label = global_dict['labels_dict']['label']
  color = ['gray',"#8c564b",'#000000',"#8c564b"]
  w = list(map(lambda x: int(x),global_dict["labels_dict"]["w"].tolist()))

  for i in range(number):
    X = np.arange(len(labels_error_array_perfold[f][i]))
    X_str = [str(i) for i in X]
    Y = labels_error_array_perfold[f][i]
    mean_err = test_loss_tensor[f][i]
    plt.xticks(X, X_str)
    plt.bar(X,Y,align='center',color=color[i],alpha=0.6,label=label[i])
    plt.plot([], [], ' ', label="Mean: "+ str(mean_err))
    plt.xlabel('Error per Intensity#')
    plt.ylabel('Error')
    plt.title('Fold: '+str(f+1)+' '+typ+' Error per Intensity in Actual Intensity Range')
    plt.legend()
    plt.xlim(xmin=0, xmax = w[i])
    plt.tight_layout()
    plt.grid()
    plt.savefig(path+"/Test/"+typ+"_ErrorPerIntensity_"+str(label[i])+"_"+"Fold:"+str(f+1)+".png")
    plt.close()

  file = open(path+"/Test/"+typ+"_Test_"+"Fold:"+str(f+1)+".txt","w+")
  file.write("Test Error fold: "+str(f+1)+' '+str(test_fold[f])+"\n")
  file.write("Scaled Test Error per fold: "+str(f+1)+' '+str(test_loss_tensor[f])+"\n")
  file.write("Parameter idx fold: "+str(f+1)+' '+str(min_param_idxArray[f])+"\n")
  file.write("Average loss "+str(mean(test_fold))+"\n")

  test_loss_tensor = [sum(i) for i in zip(*test_loss_tensor)]
  test_loss_tensor = [(i*1.0)/(f+1) for i in test_loss_tensor]

  file.write("Average loss per label: "+str(test_loss_tensor)+"\n")
  file.write("Average loss per intensity Fold:"+str(f+1)+' '+"\n")
  file.write("FOLD: "+str(f)+"\n")

  for i in range(number):
    file.write(str(labels_error_array_perfold[f][i])+"\n")
  file.write("\n")

  file.write("Predict True Array for fold:"+str(f+1)+"\n")
  file.write(str(pred_true_array[f])+"\n")

  excel_file(worksheet,f,path,pred_true_array)
  confusion(f,path,pred_true_array,label,w)
  pred_true_bar(path,f,pred_true_array,w)


  return 0


#Over 5 test folds
def plot_test(distribution_count,folds,test_fold,test_loss_tensor,path,min_param_idxArray,pred_true_array,labels_error_array,labels_error_array_perfold,typ):
  global global_dict
  number = global_dict['labels_dict']['number']
  label = global_dict['labels_dict']['label']
  color = ['gray',"#8c564b",'#000000',"#8c564b"]
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

  
  x = 1
  flatten = lambda l: [item for sublist in l for item in sublist]
  y_max = max(flatten(test_loss_tensor))+0.2

  bar_width = 0.15
  fig, ax = plt.subplots()
  ticks = [str(i+1) for i in range(folds)]
  plt.xticks(X+1.5*bar_width , ticks)

  label = global_dict['labels_dict']['label']
  for i in range(number):
    test = [j[i] for j in test_loss_tensor]
    mean_err = mean(test)
    print('mean over folds',mean_err)
    rects = plt.bar(X+i*bar_width,test,bar_width,align='center',color=color[i],alpha=0.6,label=label[i]) 
  
  plt.plot([], [], ' ', label="Mean: "+ str(mean_err))
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
  
  #plotting per intensity error averaged over after 5 folds (for each label)
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
    a = list(map(lambda x,y: x*y,distribution_count,Y))
    mean_err_label=sum(a)/sum(distribution_count)
    plt.plot([], [], ' ', label="Mean: "+ str(mean_err_label))
    plt.savefig(path+"/Test/"+typ+"_ErrorPerIntensity_"+str(label[i])+".png")
    plt.close()

    print('mean over labels',Y)

  

  pred_true_array_zipped = []
  for i in range (folds):
    pred_true = pred_true_array[i]
    pred = pred_true[0]
    true = pred_true[1]
    zipped = [(pred[p],true[p]) for p in range (len(pred))]
    pred_true_array_zipped += zipped

 # print('zipped',pred_true_array_zipped)

  file = open(path+"/Test/"+typ+"_Test.txt","w+")
  file.write("Writing for collective 5 folds \n")
  file.write("Test Error per fold "+str(test_fold)+"\n")
  file.write("Test Error per fold VAS scaled"+str(test_loss_tensor)+"\n")
  file.write("Parameter idx per fold "+str(min_param_idxArray)+"\n")
  file.write("Average loss "+str(mean(test_fold))+"\n")
  print(test_loss_tensor)
  test_loss_tensor = [sum(i) for i in zip(*test_loss_tensor)]
  test_loss_tensor = [i*1.0/folds for i in test_loss_tensor]
  file.write("Average loss per label: "+str(test_loss_tensor)+"\n")

  file.write("Average loss per label per intensity: "+str(labels_error_array)+"\n")
  length = int(len(labels_error_array)/folds)

  file.write("Average loss per intrensity per fold"+"\n")
  for f in range (folds):
    file.write("FOLD: "+str(f)+"\n")
    for i in range(number):
      file.write(str(labels_error_array_perfold[f][i])+"\n")
    file.write("\n")

  file.write('mean over folds'+ str(mean_err)+"\n")
  file.write('mean over labels'+ str(mean_err_label)+"\n")

  file.write("Predict True Array"+"\n")
  file.write(str(pred_true_array)+"\n")
  file.write("\n")

  file.write("Predict True Array Zipped"+"\n")
  for i in range (folds-1):
    file.write(str(pred_true_array_zipped[i*length:(i+1)*length])+"\n")
    file.write("\n")
  
  file.write(str(pred_true_array_zipped[(folds-1)*length:])+"\n")

  file.close()

  



