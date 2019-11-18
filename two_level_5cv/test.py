from imports import *
from skorch_module import *
from main import *


if __name__ == '__main__':
    initialise_globalDict(True)
    global global_dict
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    test = global_dict['Test']
    
    # specify which GPUs to use
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    DATA_PATH =global_dict['TEST_DATA_PATH']
    
    test_dataset = PainDataset(root_dir=DATA_PATH,
                          channels=3,
                          timeDepth=120,
                          xSize=224,
                          ySize=224,
                          turn='train',
                          files=global_dict["files"],labels_dict=global_dict["labels_dict"],test = test)

      
    
    custom_dataloader = DataLoader(test_dataset, batch_size=1)

    folds = global_dict['folds']
    # Path of the directory on which the model is saved
    MODEL_PATH_DIR = global_dict['TEST_MODEL_PATH_DIR']
    result_path = global_dict['test_result_path']

   
    test_fold_array_MSE = []
    test_fold_tensorArray_MSE = []
    test_fold_array_MAE = []
    test_fold_tensorArray_MAE = []
    pred_true_array = []

    numbers = global_dict["labels_dict"]["number"]
    w = list(map(lambda x: int(x),global_dict["labels_dict"]["w"].tolist()))
    labels_error_mse_array=[[0]*(w[num]+1) for num in range (numbers)]
    labels_error_mae_array=[[0]*(w[num]+1) for num in range (numbers)]
    distribution_count = [[0]*(w[num]+1) for num in range (numbers)]

    labels_error_mse_array_perfold = []
    labels_error_mae_array_perfold = []

    workbook = xlsxwriter.Workbook(result_path+'/arrays.xlsx')
    worksheet = workbook.add_worksheet()
    
    models = [f for f in os.listdir(MODEL_PATH_DIR)]
    #as many models as number of folds
    for MODEL_NAME in models:
      #Comment the following if only using fixed dataset for all 5models
      test_dataset = PainDataset(root_dir=DATA_PATH,
                          channels=3,
                          timeDepth=120,
                          xSize=224,
                          ySize=224,
                          turn='train',
                          files=global_dict["files"],labels_dict=global_dict["labels_dict"],test = test)

      
    
      custom_dataloader = DataLoader(test_dataset, batch_size=1)

      ###############

      test_net = initialise_model(None,0,custom_dataloader) 
      #This is just to add. the term 'module' in the params because the params have been saved on dataParallel
      state_dict = torch.load(os.path.join(MODEL_PATH_DIR,MODEL_NAME))
      from collections import OrderedDict
      new_state_dict = OrderedDict()
      for k, v in state_dict.items():
        k = k.split(".")
        #k = ["module"]+k
        k = k
        name = ".".join(k)
        new_state_dict[name] = v

      test_net.module.load_state_dict(new_state_dict)

      test_loss_MSE,test_loss_MAE,test_loss_tensor_MSE,test_loss_tensor_MAE,y_pred,y_true,\
      labels_error_mse,labels_error_mae,labels_error_mae_total,labels_error_mse_total,idx_count = test_loss_fn(test_net,test_dataset)
      
      labels_error_mse_array_perfold += [labels_error_mse] 
      labels_error_mae_array_perfold += [labels_error_mae] 

      labels_error_mse_array = list(map(lambda x_arr,y_arr: list(map(lambda x,y:x+y,x_arr,y_arr)),labels_error_mse,labels_error_mae_total))
      labels_error_mae_array = list(map(lambda x_arr,y_arr: list(map(lambda x,y:x+y,x_arr,y_arr)),labels_error_mae,labels_error_mae_total))
      distribution_count = list(map(lambda x_arr,y_arr: list(map(lambda x,y:x+y,x_arr,y_arr)),idx_count,distribution_count))

      test_fold_array_MSE += [test_loss_MSE.tolist()]
      test_fold_array_MAE += [test_loss_MAE.tolist()]
      test_fold_tensorArray_MSE  += [test_loss_tensor_MSE.tolist()]
      test_fold_tensorArray_MAE  += [test_loss_tensor_MAE.tolist()]
      print(test_loss_MSE,test_loss_MAE)
      print(test_loss_tensor_MSE,test_loss_tensor_MAE)
      print("predicit",y_pred)
      print("true label",y_true)
      print(labels_error_mse)
      print(labels_error_mae)
      pred_true_array += [[y_pred,y_true]]
      del y_pred,y_true,test_loss_tensor_MSE,test_loss_tensor_MAE
      torch.cuda.empty_cache()
      plot_test_onefold(worksheet,Ofold,folds,test_fold_array_MAE,test_fold_tensorArray_MAE,result_path,min_param_idxArray,pred_true_array,\
       labels_error_mae_array_perfold, "MAE")

      plot_test_onefold(worksheet,Ofold,folds,test_fold_array_MSE,test_fold_tensorArray_MSE,result_path,min_param_idxArray,pred_true_array,\
       labels_error_mse_array_perfold, "MSE")


    labels_error_mae_array = list(map(lambda x_arr,i_arr: list(map(lambda x,i:x/(i*folds),x_arr,i_arr)),labels_error_mae_array,distribution_count))
    labels_error_mse_array = list(map(lambda x_arr,i_arr: list(map(lambda x,i:x/(i*folds),x_arr,i_arr)),labels_error_mse_array,distribution_count))

    workbook.close()

    plot_test(folds,test_fold_array_MAE,test_fold_tensorArray_MAE,result_path,[],pred_true_array,labels_error_mae_array,labels_error_mae_array_perfold, "MAE")
    plot_test(folds,test_fold_array_MSE,test_fold_tensorArray_MSE,result_path,[],pred_true_array,labels_error_mse_array, labels_error_mse_array_perfold, "MSE")

