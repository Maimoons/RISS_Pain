from imports import *
from skorch_module import *



    
#Accuracy function for testing, called for one outer fold
def test_loss_fn(net, ds, y=None):
  global global_dict
  y_true_list = [y.tolist() for _, y in ds]
  num_seq = len(y_true_list)
  y_true = torch.FloatTensor(y_true_list)
  y_pred = torch.FloatTensor(net.predict(ds))
  if (global_dict["train_params"]["cuda"]):
    y_true.cuda()
    y_pred.cuda()

  loss_perLabel_sub = y_pred .sub(y_true) 
  loss_perLabel_sqr = torch.mul(loss_perLabel_sub,loss_perLabel_sub)  
  loss_perLabel_div = torch.mul(loss_perLabel_sqr,1.0/num_seq)
  
  loss_tensor_MSE= loss_perLabel_div.sum(0)
  #Scaling the loss/epoch to original scale
  if (global_dict["train_params"]["cuda"]):
    loss_tensor_MSE = loss_tensor_MSE.cuda()

  loss_tensor_MSE = torch.mul(loss_tensor_MSE,global_dict['labels_dict']['w'])
  
  loss_MSE = torch.mean(loss_perLabel_sqr)

  loss_perLabel_abs = loss_perLabel_sub.abs()  
  loss_tensor_MAE = torch.mul(loss_perLabel_abs,1.0/num_seq)
  loss_tensor_MAE= loss_tensor_MAE.sum(0)
  #Scaling the loss/epoch to original scale
  if (global_dict["train_params"]["cuda"]):
    loss_tensor_MAE = loss_tensor_MAE.cuda()
    
  loss_tensor_MAE= torch.mul(loss_tensor_MAE,global_dict['labels_dict']['w'])
  loss_MAE = torch.mean(loss_perLabel_abs)

  return loss_MSE,loss_MAE,loss_tensor_MSE,loss_tensor_MAE,y_pred,y_true



def initialise_model(val_dataset,idx):
  global global_dict
  network_params = global_dict["network_params"]
  train_params = global_dict["train_params"]
  
  progressbar = ProgressBar(batches_per_epoch='auto')
  epochtimer = EpochTimer()
  model = CnnRnn(pre_trained=network_params["pre_trained"], input_size=network_params["input_size"],
                   hidden_size=network_params["hidden_size"], num_layers=network_params["num_layers"][idx],
                   bias=network_params["bias"], batch_first=network_params["batch_first"],
                   dropout=network_params["dropout"][idx], bidirectional=network_params["bidirectional"],
                   global_dict=global_dict)

  torch.manual_seed(global_dict["train_params"]["seed"])
  if train_params["cuda"]:
    model.cuda()

    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(model).cuda()

  
  if train_params["cuda"]:
    torch.cuda.manual_seed(global_dict["train_params"]["seed"])

  net = NeuralNetRegressorNet(
    model,
    max_epochs=global_dict["num_epochs"],
    lr=train_params["lr"][idx],
    #make it true for random
    iterator_train__shuffle=global_dict["random"],
    batch_size=1,
    callbacks=[progressbar, epochtimer],
    warm_start=False,
    train_split=predefined_split(val_dataset),
    device = torch.device((global_dict["train_params"]["device"])),
  ) 
  net.initialize() 
  return net



# 1 is the best param
if __name__ == '__main__':
    initialise_globalDict(False)
    global global_dict
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    test = global_dict['Test']
    
    # specify which GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

      
    # train and validation datasets path
    DATA_PATH = global_dict['DATA_PATH']
    dataset = PainDataset(root_dir=DATA_PATH,
                          channels=3,
                          timeDepth=120,
                          xSize=224,
                          ySize=224,
                          turn='train',
                          files=global_dict["files"],labels_dict=global_dict["labels_dict"],test = test)

    dataloader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=1)


    #Skorch code for Cross Validation
    folds = global_dict['folds']

    # Path of the directory on which the model is saved
    MODEL_PATH_DIR = global_dict['MODEL_PATH_DIR']
    result_path = global_dict['result_path']

    dataset_len = len(dataloader)
    fold_size = int(dataset_len/folds)
    minVal_loss = float("inf")
    min_param_idx = 0
    test_fold_array_MSE = []
    test_fold_tensorArray_MSE = []
    test_fold_array_MAE = []
    test_fold_tensorArray_MAE = []
    min_param_idxArray = []
    pred_true_array = []

   
    
    #for Ofold in range(folds):
    for Ofold in range(folds):
      timestamp = time.strftime("%d%m-%H%M")
      MODEL_NAME = timestamp+"_"+'best_net.pkl'
      minVal_loss = float("inf")

      test_idx = list(range(Ofold*fold_size,Ofold*fold_size+fold_size))
      test_dataset = Subset(dataset,test_idx)
      
      train_val_idx = list(range(0,Ofold*fold_size)) +  list(range(Ofold*fold_size+fold_size,dataset_len))
      train_val_dataset = Subset(dataset,train_val_idx)
      idx = 1

      #runs train and then validation for 5 folds
      for Ifold in range(folds):
        
        if (Ifold != Ofold):
          global_dict["Ifold"] = idx-1
          val_idx = list(range(Ifold*fold_size,Ifold*fold_size+fold_size))
          train_idx = [list(range(i*fold_size,i*fold_size+fold_size)) if i!=Ifold and i!=Ofold else [] for i in range(folds)]
          flatten = lambda l: [item for sublist in l for item in sublist]
          train_idx = flatten(train_idx)
          train_dataset = Subset(dataset,train_idx)
          val_dataset = Subset(dataset,val_idx)
          num_seq_train = train_dataset.__len__()
          num_seq_val = val_dataset.__len__()
          num_seq_train_val = num_seq_train+num_seq_val
          global_dict["num_seq_train"] = num_seq_train
          global_dict["num_seq_val"] = num_seq_train
          global_dict["num_seq_train_val"] =  num_seq_train_val

          net = initialise_model(val_dataset,idx-1)
          net.fit(train_dataset,y=None)
        
          #Validates on the validation set
          val_loss =  net.history[:,'valid_loss'][-1]

          #Saving the best model
          if (val_loss < minVal_loss):
            net.save_params(f_params=os.path.join(MODEL_PATH_DIR,MODEL_NAME))
            minVal_loss = val_loss
            min_param_idx = idx-1

          #Saving epoch history
          history_file= open(result_path+"/Fold"+str(Ofold+1)+"/Fold"+str(idx)+"/fold"+str(idx)+".txt","w+")
          write_history(history_file,net,global_dict["num_epochs"])
          history_file.close()
          plot_func(net.history,global_dict["num_epochs"],result_path+"/Fold"+str(Ofold+1)+"/Fold"+str(idx))
          idx += 1

      global_dict["num_seq_train"] = global_dict["num_seq_train_val"]
      global_dict["Ifold"] = min_param_idx
      test_net = initialise_model(None,min_param_idx)
      #ask diyala if this needs to be redefined
      test_net.load_params(f_params=os.path.join(MODEL_PATH_DIR,MODEL_NAME))
      test_net.fit(train_val_dataset,y=None)

      print(min_param_idx)
      min_param_idxArray += [min_param_idx]
      test_loss_MSE,test_loss_MAE,test_loss_tensor_MSE,test_loss_tensor_MAE,y_pred,y_true = test_loss_fn(test_net,test_dataset)
      test_fold_array_MSE += [test_loss_MSE.tolist()]
      test_fold_array_MAE += [test_loss_MAE.tolist()]
      test_fold_tensorArray_MSE  += [test_loss_tensor_MSE.tolist()]
      test_fold_tensorArray_MAE  += [test_loss_tensor_MAE.tolist()]
      print(test_loss_MSE,test_loss_MAE)
      print(test_loss_tensor_MSE,test_loss_tensor_MAE)
      pred_true_array += [(y_pred.tolist(),y_true.tolist())]
    
    
    plot_test(folds,test_fold_array_MAE,test_fold_tensorArray_MAE,result_path,min_param_idxArray,pred_true_array,"MAE")
    plot_test(folds,test_fold_array_MSE,test_fold_tensorArray_MSE,result_path,min_param_idxArray,pred_true_array,"MSE")

   
