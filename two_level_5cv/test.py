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
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

      
    
    folds = 5
    # Path of the directory on which the model is saved
    MODEL_PATH_DIR = global_dict['TEST_MODEL_PATH_DIR']
    result_path = global_dict['test_result_path']

   
    test_fold_array_MSE = []
    test_fold_tensorArray_MSE = []
    test_fold_array_MAE = []
    test_fold_tensorArray_MAE = []
    pred_true_array = []

    DATA_PATH =global_dict['TEST_DATA_PATH']
    test_dataset = PainDataset(root_dir=DATA_PATH,
                          channels=3,
                          timeDepth=120,
                          xSize=224,
                          ySize=224,
                          turn='train',
                          files=global_dict["files"],labels_dict=global_dict["labels_dict"],test = test)

    models = [f for f in os.listdir(MODEL_PATH_DIR)]
    for MODEL_NAME in models:
   
      test_net = initialise_model(None,0) 
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

      test_loss_MSE,test_loss_MAE,test_loss_tensor_MSE,test_loss_tensor_MAE,y_pred,y_true = test_loss_fn(test_net,test_dataset)
      test_fold_array_MSE += [test_loss_MSE.tolist()]
      test_fold_array_MAE += [test_loss_MAE.tolist()]
      test_fold_tensorArray_MSE  += [test_loss_tensor_MSE.tolist()]
      test_fold_tensorArray_MAE  += [test_loss_tensor_MAE.tolist()]
      pred_true_array += [(y_pred.tolist(),y_true.tolist())]
      print(test_loss_MSE,test_loss_MAE)
      print(test_loss_tensor_MSE,test_loss_tensor_MAE)
      torch.cuda.empty_cache()


    plot_test(folds,test_fold_array_MAE,test_fold_tensorArray_MAE,result_path,[],pred_true_array,"MAE")
    plot_test(folds,test_fold_array_MSE,test_fold_tensorArray_MSE,result_path,[],pred_true_array,"MSE")
