from imports import *
from skorch_module import *



    
#Accuracy function for testing, called for one outer fold
#This function takes the net and the testing dataser as ds 
#and returns:
#loss_MSE,loss_MAE : average MSE and MAE loss (averaged over all labels)
#loss_tensor_MSE,loss_tensor_MAE: Error per intensity for each pain label
#y_pred_scaled,y_true_scaled: pred and true labels in org scale
#labels_error_mse,labels_error_mae: this is the total mse error, so that later could be used to calculate average over 5 folds
#labels_error_mae_total,labels_error_mse_total
#idx_count: distribution of pain intensitites in this fold

def test_loss_fn(net, ds, y=None):
  global global_dict

  #getting per intensity error
  numbers = global_dict["labels_dict"]["number"]
  w = list(map(lambda x: int(x),global_dict["labels_dict"]["w"].tolist()))

  #initialising array which saves per intensity errors
  #this is averaged for total of each intensities in each label
  labels_error_mse=[[0]*(w[num]+1) for num in range (numbers)]
  labels_error_mae=[[0]*(w[num]+1) for num in range (numbers)]

  #This only saves total for this fold, avearged over all folds later 
  labels_error_mse_total=[[0]*(w[num]+1) for num in range (numbers)]
  labels_error_mae_total=[[0]*(w[num]+1) for num in range (numbers)]

  #getting the index of each intensity in the test data
  labels_idx=[[[]]*(w[num]+1) for num in range (numbers)]
  y_true_list = [y.tolist() for _, y in ds]

  for idx in range (len(y_true_list)):
    for i in range (numbers):
        index = round(y_true_list[idx][i]*w[i])
        labels_idx[i][index] = labels_idx[i][index]+ [idx]
 
  num_seq = len(y_true_list)
  y_true = torch.FloatTensor(y_true_list)
  y_pred = torch.FloatTensor(net.predict(ds))
  #scale the pred and true value to original scale
  y_pred_scaled = [[torch.round(labels[i]*w[i]) for i in range(len(labels))] for labels in y_pred]
  y_true_scaled = [[torch.round(labels[i]*w[i]) for i in range(len(labels))] for labels in y_true]

  y_pred = torch.FloatTensor(y_pred_scaled)
  y_true = torch.FloatTensor(y_true_scaled)

  if (global_dict["train_params"]["cuda"]):
    y_true.cuda()
    y_pred.cuda()

  #MSE
  loss_perLabel_sub = y_pred.sub(y_true) 
  loss_perLabel_sqr = torch.mul(loss_perLabel_sub,loss_perLabel_sub)  
  loss_perLabel_div = torch.mul(loss_perLabel_sqr,1.0/num_seq)

  #The average loss for the test fold
  loss_tensor_MSE= loss_perLabel_div.sum(0)

  if (global_dict["train_params"]["cuda"]):
    loss_tensor_MSE = loss_tensor_MSE.cuda()

  #For each pain label then for each pain intensity: takes the correponding indices and sum and avg MSE
  for label in range (numbers):
    for num_intensity in range(w[label]+1):
        indices = labels_idx[label][num_intensity]
        sum_error = [loss_perLabel_sqr[error_idx][label] if error_idx in indices else 0 for error_idx in range (len(loss_perLabel_sqr))]
        length = sum([1 if error_idx in indices else 0 for error_idx in range (len(loss_perLabel_sqr))])

        if length!= 0:
            sum_error_avg = (sum(sum_error)/length)
        else:
            sum_error_avg = sum(sum_error)
        sum_error = sum(sum_error)
        if sum_error_avg!= 0:
          sum_error_avg = sum_error_avg.tolist()
          sum_error = sum_error.tolist()
        labels_error_mse_total[label][num_intensity] = sum_error
        labels_error_mse[label][num_intensity] = sum_error_avg

  
  #MAE 
  loss_perLabel_abs = loss_perLabel_sub.abs()  
  loss_tensor_MAE = torch.mul(loss_perLabel_abs,1.0/num_seq)
  loss_tensor_MAE= loss_tensor_MAE.sum(0)

  if (global_dict["train_params"]["cuda"]):
    loss_tensor_MAE = loss_tensor_MAE.cuda()
    
  #For each pain label then for each pain intensity: takes the correponding indices and sum and avg MSE
  for label in range (numbers):
    for num_intensity in range(w[label]+1):
        indices = labels_idx[label][num_intensity]
        sum_error = [loss_perLabel_abs[error_idx][label] if error_idx in indices else 0 for error_idx in range (len(loss_perLabel_abs))]
        length = sum([1 if error_idx in indices else 0 for error_idx in range (len(loss_perLabel_abs))])
        if length!= 0:
            sum_error_avg = (sum(sum_error)/length)
        else:
            sum_error_avg = sum(sum_error)
        sum_error = sum(sum_error)
        if sum_error_avg!= 0:
          sum_error_avg = sum_error_avg.tolist()
          sum_error = sum_error.tolist()
        labels_error_mae_total[label][num_intensity] = sum_error
        labels_error_mae[label][num_intensity] = sum_error_avg

  
  loss_MAE = torch.mean(loss_perLabel_abs)
  loss_MSE = torch.mean(loss_perLabel_sqr)

  idx_count =[[len(intensity) for intensity in label] for label in labels_idx]

  del labels_idx,loss_perLabel_abs,loss_perLabel_sqr,loss_perLabel_sub,loss_perLabel_div,y_true,y_pred


  return loss_MSE,loss_MAE,loss_tensor_MSE,loss_tensor_MAE,y_pred_scaled,y_true_scaled,labels_error_mse,labels_error_mae,labels_error_mae_total,labels_error_mse_total,idx_count










#This function initialises the model with:
#A dataset for validation and hyper parameters with gs_idx,
#Custom dataloader for later parallelization
def initialise_model(val_dataset,gs_idx,custom_dataloader):
  global global_dict
  network_params = global_dict["network_params"]
  train_params = global_dict["train_params"]
  
  progressbar = ProgressBar(batches_per_epoch='auto')
  epochtimer = EpochTimer()
  #earlystopping = EarlyStopping()

  gs_combo = global_dict['network_params']['gs_combo'][gs_idx]
  model = CnnRnn(pre_trained=network_params["pre_trained"], input_size=network_params["input_size"],
                   hidden_size=network_params["hidden_size"], num_layers=gs_combo[0],
                   bias=network_params["bias"], batch_first=network_params["batch_first"],
                   dropout=gs_combo[1], bidirectional=network_params["bidirectional"],
                   global_dict=global_dict)

  torch.manual_seed(global_dict["train_params"]["seed"])
  if train_params["cuda"]:

    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs")
      model = nn.DataParallel(model)

    model.cuda()

  
  if train_params["cuda"]:
    torch.cuda.manual_seed(global_dict["train_params"]["seed"])

  net = NeuralNetRegressorNet(
    model,
    max_epochs=global_dict["num_epochs"],
    lr=train_params["lr"],
    iterator_train__shuffle=global_dict["random"],
    batch_size=1,
    callbacks=[progressbar, epochtimer],
    warm_start=False,
    train_split=predefined_split(val_dataset),
    device = torch.device((global_dict["train_params"]["device"])),
    #iterator_train = custom_dataloader,
    #iterator_valid = custom_dataloader,
  ) 
  net.initialize() 
  return net







#For parallel code for later
def my_collate(batch):
    print('custom collate')
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]







if __name__ == '__main__':
    initialise_globalDict(False)
    global global_dict
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    test = global_dict['Test']

    # specify which GPUs to use
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,1,2,0'
    flatten = lambda l: [item for sublist in l for item in sublist]

      
    # train and validation datasets path
    DATA_PATH = global_dict['DATA_PATH']
    dataset = PainDataset(root_dir=DATA_PATH,
                          channels=3,
                          timeDepth=120,
                          xSize=224,
                          ySize=224,
                          turn='train',
                          files=global_dict["files"],labels_dict=global_dict["labels_dict"],test = test)

    #custom_dataloader = DataLoader(dataset, batch_size=4,shuffle=False, collate_fn=my_collate, num_workers=4)
    custom_dataloader = DataLoader(dataset, batch_size=1)
    dataset_len = len(custom_dataloader)
    print("dataset length",dataset_len)
    
    folds = global_dict['folds']
    # Path of the directory on which the model is saved
    MODEL_PATH_DIR = global_dict['MODEL_PATH_DIR']
    result_path = global_dict['result_path']
    numbers = global_dict["labels_dict"]["number"]
    w = list(map(lambda x: int(x),global_dict["labels_dict"]["w"].tolist()))
    gs_combo = global_dict['network_params']['gs_combo']
    all_idx_array = global_dict['all_idx_array']

 
    #initialose min loss
    minVal_loss = float("inf")
    min_param_idx = 0
    test_fold_array_MSE = []
    test_fold_tensorArray_MSE = []
    test_fold_array_MAE = []
    test_fold_tensorArray_MAE = []
    min_param_idxArray = []
    pred_true_array = []
    
    #For accumulatin the total MSE and MAE error
    labels_error_mse_array=[[0]*(w[num]+1) for num in range (numbers)]
    labels_error_mae_array=[[0]*(w[num]+1) for num in range (numbers)]
    distribution_count = [[0]*(w[num]+1) for num in range (numbers)]

    labels_error_mse_array_perfold = []
    labels_error_mae_array_perfold = []


    workbook = xlsxwriter.Workbook(result_path+'/arrays.xlsx')
    worksheet = workbook.add_worksheet()

    

    #Outer fold run 5 level outer CV for test dataset
    for Ofold in range(folds):
      timestamp = time.strftime("%d%m-%H%M")
      MODEL_NAME = timestamp+"_"+'best_net.pkl'
      global_dict['MODEL_NAME'] = MODEL_NAME
      minVal_loss = float("inf")
      test_idx = all_idx_array[Ofold]
      #Picks the ith fold as the test fold
      test_dataset = Subset(dataset,test_idx)

      train_val_idx = flatten(all_idx_array[:Ofold]+all_idx_array[Ofold+1:])
      train_val_dataset = Subset(dataset,train_val_idx)
      
      idx = 0

      #Inner fold run 4 level inner CV for train and val dataset
      for Ifold in range(folds):
        
        if (Ifold != Ofold):
          global_dict["Ifold"] = idx
          
          #Picks the jth fold from the remaining folds as validation fold
          val_idx = all_idx_array[Ifold]
          
          #Puts the rest of the folds as training folds
          train_idx = [all_idx_array[i] if i!=Ifold and i!=Ofold else [] for i in range(folds)]
          train_idx = flatten(train_idx)       
          train_dataset = Subset(dataset,train_idx)
          val_dataset = Subset(dataset,val_idx)
          num_seq_train = train_dataset.__len__()
          num_seq_val = val_dataset.__len__()
          num_seq_train_val = num_seq_train+num_seq_val

          global_dict["num_seq_train"] = num_seq_train
          global_dict["num_seq_val"] = num_seq_val
          global_dict["num_seq_train_val"] =  num_seq_train_val
          idx += 1

          print("test index",test_idx)
          print("val idx",num_seq_val,val_idx)
          print("train idx",num_seq_train,train_idx)
          
          #Runs over different combinations for Grid Search for each inner fold
          for gs_idx in range(len(gs_combo)):
            global_dict['gs_idx'] = gs_idx
            net = initialise_model(val_dataset,gs_idx,custom_dataloader)
            net.fit(train_dataset,y=None)
        
            #Validates on the validation set, last epochs val error
            val_loss =  net.history[:,'valid_loss'][-1]
            #Saving the best model
            if (val_loss < minVal_loss):
              net.save_params(f_params=os.path.join(MODEL_PATH_DIR,MODEL_NAME))
              minVal_loss = val_loss
              min_param_idx = gs_idx
              print('min param',min_param_idx)

            #Saving epoch history
            history_file= open(result_path+"/Fold"+str(Ofold+1)+"/Fold"+str(idx)+"/param_"+str(gs_idx)+"/fold"+str(idx)+".txt","w+")
            write_history(history_file,net,global_dict["num_epochs"])
            history_file.close()
            plot_func(net.history,global_dict["num_epochs"],result_path+"/Fold"+str(Ofold+1)+"/Fold"+str(idx)+"/param_"+str(gs_idx))
            torch.cuda.empty_cache()
        
      
      global_dict["num_seq_train"] = global_dict["num_seq_train_val"]
      global_dict["gs_idx"] = min_param_idx
      global_dict["train_val"] = True
      min_param_idxArray += [min_param_idx]

      #Reintialises the model for testing
      test_net = initialise_model(test_dataset,min_param_idx,custom_dataloader)
      #Loads the best model saved from all combination of grid search
      test_net.load_params(f_params=os.path.join(MODEL_PATH_DIR,MODEL_NAME))
      #Retrains on the combined train and val dataset
      test_net.fit(train_val_dataset,y=None)

      global_dict['MODEL_NAME'] = "_test:"+str(Ofold)+"_gs:"+str(min_param_idx)+'_'+MODEL_NAME
      MODEL_NAME = global_dict['MODEL_NAME']
      #Saved the retrained model in Test dir in models
      test_net.save_params(f_params=os.path.join(MODEL_PATH_DIR+"/Test",MODEL_NAME))
      #Saves result for this training
      history_file= open(result_path+"/Fold"+str(Ofold+1)+"/Fold"+str(idx+1)+"/fold_test"+str(idx+1)+".txt","w+")
      write_history(history_file,net,global_dict["num_epochs"])
      history_file.close()
      plot_func(net.history,global_dict["num_epochs"],result_path+"/Fold"+str(Ofold+1)+"/Fold"+str(idx+1))
      global_dict["train_val"] = False

      #Test on the test dataset
      test_loss_MSE,test_loss_MAE,test_loss_tensor_MSE,test_loss_tensor_MAE,y_pred,y_true,\
      labels_error_mse,labels_error_mae,labels_error_mae_total,labels_error_mse_total,idx_count = test_loss_fn(test_net,test_dataset)
      
      labels_error_mse_array_perfold += [labels_error_mse] 
      labels_error_mae_array_perfold += [labels_error_mae] 

      labels_error_mse_array = list(map(lambda x_arr,y_arr: list(map(lambda x,y:x+y,x_arr,y_arr)),labels_error_mse_array,labels_error_mse_total))
      labels_error_mae_array = list(map(lambda x_arr,y_arr: list(map(lambda x,y:x+y,x_arr,y_arr)),labels_error_mae_array,labels_error_mae_total))
      distribution_count = list(map(lambda x_arr,y_arr: list(map(lambda x,y:x+y,x_arr,y_arr)),idx_count,distribution_count))

      test_fold_array_MSE += [test_loss_MSE.tolist()]
      test_fold_array_MAE += [test_loss_MAE.tolist()]
      test_fold_tensorArray_MSE  += [test_loss_tensor_MSE.tolist()]
      test_fold_tensorArray_MAE  += [test_loss_tensor_MAE.tolist()]
      pred_true_array += [[y_pred,y_true]]

      print('idx count',idx_count)
      print(test_loss_MSE,test_loss_MAE)
      print(test_loss_tensor_MSE,test_loss_tensor_MAE)
      print(labels_error_mse)
      print(labels_error_mae)
      del y_pred,y_true,test_loss_tensor_MSE,test_loss_tensor_MAE
      
      plot_test_onefold(worksheet,Ofold,folds,test_fold_array_MAE,test_fold_tensorArray_MAE,result_path,min_param_idxArray,pred_true_array,\
       labels_error_mae_array_perfold, "MAE")

      plot_test_onefold(worksheet,Ofold,folds,test_fold_array_MSE,test_fold_tensorArray_MSE,result_path,min_param_idxArray,pred_true_array,\
       labels_error_mse_array_perfold, "MSE")
      #workbook.close()
     
    #the overall error per inetnsity over all 5 test folds
    labels_error_mae_array = list(map(lambda x_arr,i_arr: list(map(lambda x,i:x/(i),x_arr,i_arr)),labels_error_mae_array,distribution_count))
    labels_error_mse_array = list(map(lambda x_arr,i_arr: list(map(lambda x,i:x/(i),x_arr,i_arr)),labels_error_mse_array,distribution_count))
    
    workbook.close()
    plot_test(distribution_count,folds,test_fold_array_MAE,test_fold_tensorArray_MAE,result_path,min_param_idxArray,pred_true_array,labels_error_mae_array,labels_error_mae_array_perfold, "MAE")
    plot_test(distribution_count,folds,test_fold_array_MSE,test_fold_tensorArray_MSE,result_path,min_param_idxArray,pred_true_array,labels_error_mse_array, labels_error_mse_array_perfold, "MSE")

   
