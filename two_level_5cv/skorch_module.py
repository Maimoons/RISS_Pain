from imports import *

#Called for every sequence, overwriting the default loss function
class NeuralNetRegressorNet(NeuralNetRegressor):
  def get_loss(self, y_pred, y_true, *args, **kwargs):
        global global_dict
        loss_tensor_train = global_dict["loss_tensor_train"]
        loss_tensor_val = global_dict["loss_tensor_val"]
        training = kwargs['training']
        Ifold = global_dict["Ifold"]
        regularization = global_dict["network_params"]["lambda"][Ifold] * sum([(w**2).sum() for w in self.module_.parameters()]) 
        if (global_dict["train_params"]["cuda"]):
            y_true = y_true.cuda()

        #if MSE
        loss_perLabel_sub = y_pred.sub(y_true)
        loss_perLabel_sqr = torch.mul(loss_perLabel_sub,loss_perLabel_sub)  
        loss_perLabel = loss_perLabel_sqr
        loss = torch.mean(loss_perLabel_sqr)


        if global_dict['train_params']['training_loss_func'] == 'custom':
            loss_perLabel = (global_dict["train_params"]["custom_loss_alpha"] * loss_perLabel_sqr) + ((1 - global_dict["train_params"]["custom_loss_alpha"]) * torch.std(y_pred))
            loss = (global_dict["train_params"]["custom_loss_alpha"] * loss) + ((1 - global_dict["train_params"]["custom_loss_alpha"]) * torch.std(y_pred))

        if global_dict['train_params']['regularization']:
            loss += regularization

        if (training):
          loss_tensor_train += loss_perLabel
        else:
          loss_tensor_val += loss_perLabel

        global_dict["loss_tensor_train"] = loss_tensor_train
        global_dict["loss_tensor_val"] =  loss_tensor_val
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
        loss_tensor_train = global_dict["loss_tensor_train"]
        loss_tensor_val = global_dict["loss_tensor_val"]
        num_seq_train = global_dict["num_seq_train"]
        num_seq_val = global_dict["num_seq_val"]

        loss_tensor = loss_tensor_train
        loss_tensor= torch.mul(loss_tensor,1.0/num_seq_train)
        loss_tensor_train = torch.zeros([1,global_dict['labels_dict']['number']],device=torch.device(global_dict["train_params"]["device"]))
        string = 'separate_loss_train'
        #Scaling the loss/epoch to original scale
        loss_tensor = torch.flatten(loss_tensor)
        w = global_dict['labels_dict']['w']
        if (global_dict["train_params"]["cuda"]):
            loss_tensor = loss_tensor.cuda()
            w = w.cuda()

        loss_tensor = torch.mul(loss_tensor,w)

        net.history.record(string, loss_tensor)       
  
    
        loss_tensor = loss_tensor_val
        loss_tensor= torch.mul(loss_tensor,1.0/num_seq_val)
        loss_tensor_val = torch.zeros([1,global_dict['labels_dict']['number']],device=torch.device(global_dict["train_params"]["device"]))
        string = 'separate_loss_val'
        #Scaling the loss/epoch to original scale
        loss_tensor = torch.flatten(loss_tensor)
        if (global_dict["train_params"]["cuda"]):
            loss_tensor = loss_tensor.cuda()
        loss_tensor = torch.mul(loss_tensor.cuda(),w)
        net.history.record(string, loss_tensor) 
        net.history.record('dur', time.time() - self.epoch_start_time_)
        global_dict["loss_tensor_train"] = loss_tensor_train
        global_dict["loss_tensor_val"] =  loss_tensor_val

