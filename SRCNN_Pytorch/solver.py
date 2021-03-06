from __future__ import division
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import scipy.misc
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

    After train() method is called, the self.model will be the best model on 
    validation set. The best model is saved into trained_model.pt, which is used
    for the testing time. 

    For statistics, loss history, avr_train_psnr history, andavr_val_psnr history
    are also saved. 
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
        - batch_size: batch size for train phase
        - optimizer: update rule for model parameters
        - loss_fn: loss function for the model
        - verbose: print statistics
        - print_every: period of statistics printing
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

        self._reset()

    def _reset(self):
        """ Initialize some book-keeping variable, dont call it manually"""
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model = self.model.cuda()
        self.hist_train_psnr = []
        self.hist_val_psnr = []
        self.hist_loss = []
    
    def _epoch_step(self, epoch):
        """
        Perform 1 training epoch
        """
        for i, (input_batch, label_batch) in enumerate(self.dataloaders['train']):
            #Wrap with torch Variable
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)

            #zero the grad
            self.optimizer.zero_grad()

            # Forward
            output_batch = self.model(input_batch)
            loss = self.loss_fn(output_batch, label_batch)

            # save statistic
            self.hist_loss.append(loss.data[0])
            
            if self.verbose:
                if i%self.print_every== 0:
                    print('epoch %5d iter %5d, loss %.5f' \
                            %(epoch, i, loss.data[0]))
            
            # Backward + update
            loss.backward()
            self.optimizer.step()

    def _wrap_variable(self, input_batch, label_batch, use_gpu):
        if use_gpu:
            input_batch, label_batch = (Variable(input_batch.cuda()),
                                        Variable(label_batch.cuda()))
        else:
            input_batch, label_batch = (Variable(input_batch),
                                        Variable(label_batch))
        return input_batch, label_batch
    
    def _comput_PSNR(self, imgs1, imgs2):
        """Compute PSNR for gray scale"""
        # TODO: check for 3 channel image
        N = imgs1.size()[0]
        imdiff = imgs1 - imgs2
        imdiff = imdiff.view(N, -1)
        rmse = torch.sqrt(torch.mean(imdiff**2, dim=1))
        psnr = 20*torch.log(255/rmse)/math.log(10) # psnr = 20*log10(255/rmse)
        psnr =  torch.sum(psnr)
        return psnr

                
    def _check_PSNR(self, dataset, is_test=False, batch_size=100):
        """
        Compute the PSNR for the dataset at training, validation, and testing time
        For training, and validation, the images are cropped in to multiple subimages, 
        but for the testing we use the original size of images.
        The cropped-input, output, label images are save on Result/
        """
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
        
        avr_psnr = 0
        for batch, (input_batch, label_batch) in enumerate(dataloader):
            input_batch, label_batch = self._wrap_variable(input_batch,
                                                           label_batch,
                                                           self.use_gpu)
            
            output_batch = self.model(input_batch)
            
            output = output_batch.clone().data
            label = label_batch.clone().data
            
            output = output.squeeze(dim=1)
            label = label.squeeze(dim=1)
            
            # use original image size for testing
            if is_test:
                inp = input_batch.clone().data
                inp = inp.squeeze(dim=1)
                save_output = output.cpu().numpy()
                save_label = label.cpu().numpy()
                save_input = inp.cpu().numpy()
                
                # crop input
                offset = self.model.offset
                save_input = save_input[:, offset:-offset, offset:-offset]
                
                scipy.misc.imsave('Result/output_{}.png'.format(batch), 
                                  save_output[0])
                scipy.misc.imsave('Result/label_{}.png'.format(batch), 
                                  save_label[0])
                scipy.misc.imsave('Result/input_{}.png'.format(batch),
                                  save_input[0])
            
            psnr = self._comput_PSNR(output, label)
            avr_psnr += psnr
            
        epoch_size = len(dataset)
        avr_psnr /= epoch_size

        return avr_psnr

     
    def train(self):
        """
        Load data form self.datasets and train the network.
        The optimal model is saved in traned_model.pt
        """

        # load data
        train_loader = DataLoader(self.datasets['train'], batch_size=self.batch_size,
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(self.datasets['val'], batch_size=self.batch_size,
                                shuffle=False, num_workers=4)
        self.dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # capture best model
        best_val_psnr = -1
        best_model_state = self.model.state_dict()

        # Train the model
        for epoch in range(self.num_epochs):
            self._epoch_step(epoch)
            
            # capture running PSNR on train and val dataset
            train_psnr = self._check_PSNR(self.datasets['train'])
            val_psnr = self._check_PSNR(self.datasets['val'])
            self.hist_train_psnr.append(train_psnr)
            self.hist_val_psnr.append(val_psnr)

            if best_val_psnr < val_psnr:
                best_val_psnr = val_psnr
                best_model_state = self.model.state_dict()

        # save the best model to self.model
        self.model.load_state_dict(best_model_state)
        # write the model to hard-disk for testing
        torch.save(self.model, 'trained_model.pt')

    def test(self, dataset):
        """
        Load the model stored in train_model.pt from training phase,
        then return the average PNSR on test samples.
        The cropped-input, output, label images are save on Result/
        """
        try:
            self.model = torch.load('trained_model.pt')
            test_psnr = self._check_PSNR(dataset, is_test=True, batch_size=1)
            print('Average test PSNR: %.2fdB' %test_psnr)
            
        except:
            print('Exception!!!')
            print('Cannot load trained_model.pt, you probably did not train the model.')

