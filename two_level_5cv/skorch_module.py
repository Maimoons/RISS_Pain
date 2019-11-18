from imports import *

#Called for every sequence, overwriting the default loss function
class NeuralNetRegressorNet(NeuralNetRegressor):
  def get_loss(self, y_pred, y_true, *args, **kwargs):
        global global_dict
        training = kwargs['training']
        gs_idx = global_dict["gs_idx"]
        #lambda is the 3rd param in gs
        regularization = global_dict["network_params"]["gs_combo"][gs_idx][2] * sum([(w**2).sum() for w in self.module_.parameters()]) 
        number = global_dict['labels_dict']['number']

        w = global_dict['labels_dict']['w']
        if (global_dict["train_params"]["cuda"]):
            y_true = y_true.cuda()
            y_pred = y_pred.cuda()
            w = w.cuda()


        #if MSE
        loss_perLabel_sub = y_pred.sub(y_true)
        loss_perLabel_sqr = torch.mul(loss_perLabel_sub,loss_perLabel_sub)  
        loss_perLabel = loss_perLabel_sqr
        #mean for average loss over labels e.g VAS
        loss = torch.mean(loss_perLabel_sqr)


        if global_dict['train_params']['training_loss_func'] == 'custom':
            loss_perLabel = (global_dict["train_params"]["custom_loss_alpha"] * loss_perLabel_sqr) + ((1 - global_dict["train_params"]["custom_loss_alpha"]) * torch.std(y_pred))
            loss = (global_dict["train_params"]["custom_loss_alpha"] * loss) + ((1 - global_dict["train_params"]["custom_loss_alpha"]) * torch.std(y_pred))

        if global_dict['train_params']['regularization']:
            loss += regularization

        #Calculating just MSE and MAE without regularization in the original intensity range
        #for i in range (number):
        y_true = y_true*w
        y_pred = y_pred*w

        loss_perLabel_sub = y_pred.sub(y_true)

        if (training):
          global_dict["loss_tensor_train_MAE"] += loss_perLabel_sub.abs()
          global_dict["loss_tensor_train_MSE"] += torch.mul(loss_perLabel_sub,loss_perLabel_sub) 
        
        else:
          global_dict["loss_tensor_val_MAE"] +=  loss_perLabel_sub.abs()
          global_dict["loss_tensor_val_MSE"] +=  torch.mul(loss_perLabel_sub,loss_perLabel_sub) 

        torch.cuda.empty_cache()
        
        
        return loss
        
        


class EpochTimer(Callback):
    def __init__(self, **kwargs):
        super(EpochTimer, self).__init__(**kwargs)

        self.epoch_start_time_ = None

    def on_epoch_begin(self, net, **kwargs):
        self.epoch_start_time_ = time.time()

        
    def on_epoch_end(self, net,**kwargs):
        global global_dict
        num_seq_train = global_dict["num_seq_train"]
        num_seq_val = global_dict["num_seq_val"]

        #Avearges the sum of the loss over total number of sequences
        loss_tensor_train_MAE = torch.mul(global_dict["loss_tensor_train_MAE"],1.0/num_seq_train)
        loss_tensor_val_MAE = torch.mul(global_dict["loss_tensor_val_MAE"],1.0/num_seq_val)

        loss_tensor_train_MSE = torch.mul(global_dict["loss_tensor_train_MSE"],1.0/num_seq_train)
        loss_tensor_val_MSE = torch.mul(global_dict["loss_tensor_val_MSE"],1.0/num_seq_val)
            
        
        net.history.record('loss_tensor_train_MAE', loss_tensor_train_MAE)     
        net.history.record('loss_tensor_train_MSE', loss_tensor_train_MSE)       
        net.history.record('loss_tensor_val_MAE', loss_tensor_val_MAE) 
        net.history.record('loss_tensor_val_MSE', loss_tensor_val_MSE) 

        #Reset so the error can be accumulated here next epoch
        global_dict["loss_tensor_train_MAE"] = torch.zeros([1,global_dict['labels_dict']['number']],device=torch.device(global_dict["train_params"]["device"]))
        global_dict["loss_tensor_val_MAE"] =  torch.zeros([1,global_dict['labels_dict']['number']],device=torch.device(global_dict["train_params"]["device"]))
        global_dict["loss_tensor_train_MSE"] = torch.zeros([1,global_dict['labels_dict']['number']],device=torch.device(global_dict["train_params"]["device"]))
        global_dict["loss_tensor_val_MSE"] = torch.zeros([1,global_dict['labels_dict']['number']],device=torch.device(global_dict["train_params"]["device"]))

        net.history.record('dur', time.time() - self.epoch_start_time_)
        valid_loss = net.history[:,'valid_loss']
        #if (global_dict["train_val"]):
            #print('len'+str(len(valid_loss)))
     
        gs_idx = global_dict['gs_idx'] 
        MODEL_PATH_DIR = global_dict['MODEL_PATH_DIR'] 
        MODEL_NAME = global_dict['MODEL_NAME'] 

        #Checking if the validation loss has stopped improving for 2 epochs on a threshold of 0.005
        if (global_dict["train_val"]):
            if (len(valid_loss))> 3:
                if (valid_loss[-1]-valid_loss[-2] >0.005 and valid_loss[-1]-valid_loss[-3]>0.005):
                    net.save_params(f_params=os.path.join(MODEL_PATH_DIR+"/Test","_early:"+str(len(valid_loss))+"_"+MODEL_NAME))




