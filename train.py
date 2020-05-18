from load_data import get_loaders
from config import get_config
import torch
from mixup import *
import numpy as np
from tqdm.auto import tqdm
import torch.backends.cudnn as cudnn
from utils.optimizer_wrapper import *
from run import train as run_train
from run import test as run_test
from run import save_checkpoint
from utils.logger import *
from utils.visualize import Visualizer
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CyclicLR
class Train :
    def __init__(self):
        opt=get_config()
        self.opt=opt
        self.device=opt.device
        self.train_loader,self.val_loader=get_loaders(opt.dataroot,opt.val_batch_size,opt.train_batch_size,opt.input_size,opt.workers,opt.num_nodes, opt.local_rank)
        print(len(self.train_loader))
        self.epochs_per_step=int(len(self.train_loader.dataset)/opt.train_batch_size)
        if(opt.back_bone=="mobiv3"):
            from backbone.mobilev3.mobilenetV3 import mobilenetv3
            self.model=mobilenetv3(input_size=opt.input_size,num_classes=2,small=False,get_weights=False).to(self.device)
            print("use mobiv3")
        elif(opt.back_bone=="efficient"):
            from backbone.efficient.efficientNet import EfficientNet
            self.model=EfficientNet.from_name('efficientnet-b0',image_size=opt.input_size,num_classes=opt.num_classes).to(self.device)
            print("use efficient")
           

        self.lr=opt.learning_rate
        if(opt.loss=="cross_entropy"):
            from utils.cross_entropy import CrossEntropyLoss
            self.criterion=CrossEntropyLoss().to(self.device)
        if(self.device=='cuda'):
            cudnn.enabled = True
            print("cudnn enable")
        self.epochs=opt.epochs
        
        num_parameters = sum([l.nelement() for l in self.model.parameters()])
        print("num parameters",num_parameters)

        self.optim,self.mixup = self.init_optimizer_and_mixup()
        self.vis=Visualizer()
        
    
    def init_optimizer_and_mixup(self,optim_state_dict=None):
        optimizer_class = torch.optim.SGD
        optimizer_params = {"lr": self.opt.learning_rate, "momentum": self.opt.momentum, "weight_decay": self.opt.decay,
                            "nesterov": True}

        if self.opt.sched == 'clr':
            scheduler_class = CyclicLR

            scheduler_params = {"base_lr": self.opt.min_lr, "max_lr": self.opt.max_lr,
                                "step_size_up": self.epochs_per_step * len(self.train_loader), "mode": self.opt.mode,
                                "last_epoch": self.opt.start_step - 1}

        optim = OptimizerWrapper(self.model, optimizer_class=optimizer_class, optimizer_params=optimizer_params,
                                optimizer_state_dict=optim_state_dict, scheduler_class=scheduler_class,
                                scheduler_params=scheduler_params, use_shadow_weights=self.opt._dtype == torch.float16)
        mixup_start = len(self.train_loader) * self.opt.mixup_warmup
        mixup_nr = len(self.train_loader) * (self.opt.epochs - self.opt.mixup_warmup)
        mixup = MixupScheduled(start_gamma=0, stop_gamma=self.opt.mixup, wait_steps=mixup_start, nr_steps=mixup_nr,
                            start_step=self.opt.start_step, num_classes=self.opt.num_classes, smooth_eps=self.opt.smooth_eps)
        return optim, mixup

    def train_network(self,resume=False):
        csv_logger = CsvLogger(filepath=self.opt.save_model)
        if(not resume):
            best_test = 0
            
            start_epoch=self.opt.start_epoch
        else:
            checkpoint_path = os.path.join(self.opt.resume, 'checkpoint{}.pth.tar'.format(self.opt.local_rank))
            csv_path = os.path.join(self.opt.resume, 'results{}.csv'.format(self.opt.local_rank))
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=self.opt.device)
            start_epoch = checkpoint['epoch']
            start_step = len(self.train_loader) * start_epoch
            self.optim, self.mixup = self.init_optimizer_and_mixup(checkpoint['optimizer'])
            best_test = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    
        for epoch in range(start_epoch, self.opt.epochs + 1):
            print("-------------------------",epoch,"--------------------------------")
            
            train_loss, train_accuracy1, train_accuracy5, = run_train(self.model, self.train_loader, self.mixup, epoch, self.optim, self.criterion,self.device, self.opt._dtype, self.opt.train_batch_size,log_interval=2)
            test_loss, test_accuracy1, test_accuracy5 = run_test(self.model, self.val_loader, self.criterion, self.device, self.opt._dtype)
            
            self.optim.epoch_step()
            self.vis.plot_curves({'train_loss': train_loss}, iters=epoch, title='train loss',
                                xlabel='epoch', ylabel='train loss')
            self.vis.plot_curves({'train_acc': train_accuracy1}, iters=epoch, title='train acc',
                                xlabel='epoch', ylabel='train acc')
            self.vis.plot_curves({'val_loss': test_loss}, iters=epoch, title='val loss',
                                xlabel='epoch', ylabel='val loss')
            self.vis.plot_curves({'val_acc': test_accuracy1}, iters=epoch, title='val acc',
                                xlabel='epoch', ylabel='val acc')

            csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy1, 'val_error5': 1 - test_accuracy5,
                            'val_loss': test_loss, 'train_error1': 1 - train_accuracy1,
                            'train_error5': 1 - train_accuracy5, 'train_loss': train_loss})
            save_checkpoint({'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'best_prec1': best_test,
                            'optimizer': self.optim.state_dict()}, test_accuracy1 > best_test, filepath=self.opt.save_model,
                            local_rank=self.opt.local_rank)
            # TODO: save on the end of the cycle

            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)
            print("memory gpu use : ",mem)
            if test_accuracy1 > best_test:
                best_test = test_accuracy1
    

if __name__ == "__main__":
    train=Train()
    train.train_network(False)


