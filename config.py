from easydict import EasyDict as Edict 
import torch
def get_config():
    conf=Edict()
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # conf.device=torch.device("cpu")
    conf.dataroot="data"
    conf.val_batch_size=64
    conf.train_batch_size=64
    conf.input_size=112
    conf.workers=4
    conf.num_nodes=2 
    conf.local_rank=-1
    # conf.back_bone="mobiv3"
    conf.loss="cross_entropy"
    conf.back_bone="efficient"
    conf.learning_rate = 0.01
    conf.start_step=0
    
    conf.save_model=conf.back_bone
    conf.resume=conf.save_model
    conf.pretrained=False
    conf.num_classes=7
    
    # mixup
    conf.mixup=0.1
    conf.step=40 # decrease learning rate each time
    conf.smooth_eps=0.1
    conf.mixup_warmup=455
    
    #optimizer_params
    conf.min_lr=2.5e-6
    conf.max_lr=0.225
    conf.sched = "clr"
    conf.decay=2.5e-7
    conf.momentum=0.9
    
    
    #epochs
    conf.start_epoch=0
    conf.epochs=100000
    conf._dtype = torch.float16
    
    conf.mode="triangular2"

    

    
    



    return conf