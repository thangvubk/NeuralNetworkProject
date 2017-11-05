import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_loader import SRCNN_dataset
from model import SRCNN

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training super resolution
    The Solver accepts both training and validation data label so it can 
    periodically check the PSNR on training and validation.
    
    To train a model, you will first construct a Solver instance, pass the model,
    datasets, and various option (optimizer, loss_fn, batch_size, etc) to the
    constructor.

    After train() method return, the self.model will be the best model on 
    validation set. For statistics, loss history, avr_train_psnr history,
    avr_val_psnr history is saved into ...
    """
    def __init__(self, model, datasets, **kwargs):
        """
        Construct a new Solver instance

        Required arguments
        - model: a torch nn module describe the neural network architecture
        - datasets: a dictionary of training and validation data, which are
        used to pass to DataLoader.
            'train': training dataset
            'val': validation dataset

        Optional arguments:
        - num_epochs: number of epochs to run during training
        - batch_size: batch size ...

        """
        self.model = model
        self.datasets = datasets
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 128)
        self.optimizer = kwargs.pop('optimizer', 
                                    optim.Adam(model.parameters(), lr=1e-4))
        self.loss_fn = kwargs.pop('loss_fn', nn.MSELoss())
        self.verbose = kwargs.pop('verbose', False)
        self.print_every = kwargs.pop('print_every', 10)

        #
        self._reset()

    def _reset(self):
        self.use_gpu = torch.cuda.is_available()
        self.hist_train_psnr = []
        self.hist_val_psnr = []
        self.hist_loss = []

    def __str__(self):
        strs= 'SOLVER CONFIGS'
        strs += 'Loss function:' + str(self.loss_fn)
        #print('Optimizer:', self.optimizer)
        #print('Num epochs:', self.num_epochs)
        #print('Batch size:', self.batch_size)
        return strs
    
    def _epoch_step(self):
        """
        Perform 1 training and validation epoch, capture history of loss, 
        avr_train_psnr, avr_val_psnr
        """
        epoch_avr_psnr = {} # dictionary keeps psnr of train and val
        for phase in ['train', 'val']:
            if phase == 'train':
                self.model.train(True) # torch API, set model to train mode
            else:
                self.model.train(False) # torch API, set model to evaluation mode

            for i, (input_batch, label_batch) in enumerate(self.dataloaders[phase]):
                #Wrap with torch Variable
                input_batch, label_batch =  (Variable(input_batch), 
                                             Variable(label_batch))
                
                #zero the grad
                self.optimizer.zero_grad()

                # Forward
                output_batch = self.model(input_batch)
                
                # compute and update epoch avr psnr
                batch_avr_psnr = self.check_avr_PSNR(output_batch, label_batch)
                # use get for safe increment
                epoch_avr_psnr[phase] = epoch_avr_psnr.get(phase, 0) + batch_avr_psnr

                loss = self.loss_fn(output_batch, label_batch)
                
                if self.verbose:
                    if i%self.print_every== 0:
                        print('iter %5d, loss %.5f' \
                                %(i, loss.data[0]))
                
                # Backward + update
                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            # compute average psnr
            epoch_size = len(self.datasets[phase])
            epoch_avr_psnr[phase] = epoch_avr_psnr[phase]*self.batch_size/epoch_size
        
        # capture psnr
        self.hist_train_psnr.append(epoch_avr_psnr['train'])
        self.hist_val_psnr.append(epoch_avr_psnr['val'])
    
    def check_avr_PSNR(self, output_batch, label_batch):
        # output_batch: shape (N, C, H1, W1)
        # label_batch: shape (N, C, H2, W2)
        # H2 = H1 + offset
        # W2 = W1 + offset

        N = output_batch.size()[0]
        
        # Swap to numpy array
        output = output_batch.clone().data.numpy()
        label = label_batch.clone().data.numpy()

        # change to numpy image (N, H, W, C)
        output = output.transpose(0, 2, 3, 1)
        label = label.transpose(0, 2, 3, 1)
         
        # Compute PSNR for gray scale
        # TODO: check for 3 channel image
        imdiff = output - label
        imdiff = imdiff.reshape(N, -1)
        rmse = np.sqrt(np.linalg.norm(imdiff, axis=1))
        psnr = 20*np.log10(255/rmse)
        avr_psnr = np.mean(psnr)
        return avr_psnr



        
    def train(self):
        """
        Train the network

        Args:
            - dataloader: used to load minibatch
            - model: model for compute output
            - loss_fn: loss function
            - optimizer: weight update scheme
            - num_epochs: number of epochs
        """

        # load data
        #dataset = SRCNN_dataset(data_config, transforms.ToTensor())
        train_loader = DataLoader(self.datasets['train'], batch_size=self.batch_size,
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(self.datasets['val'], batch_size=self.batch_size,
                                shuffle=False, num_workers=4)
        self.dataloaders = {
            'train': train_loader,
            'val': val_loader
        }

        # Train the model
        for epoch in range(self.num_epochs):
            self._epoch_step()
           
