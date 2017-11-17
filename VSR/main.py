import argparse
import sys
import os
from SR_datasets import SRCNN_dataset, SR_dataset
from model import SRCNN, SRCNN_proposed
from solver import Solver

description='SRCNN-pytorch implementation'

parser = argparse.ArgumentParser(description=description)


parser.add_argument('phase', metavar='PHASE', type=str,
                    help='train or test or both')
parser.add_argument('-m', '--model', metavar='M', type=str, default='SRCNN_proposed',
                    help='network architecture')
parser.add_argument('-c', '--scale', metavar='S', type=int, default=3, 
                    help='interpolation scale')
parser.add_argument('--train-path', metavar='PATH', type=str, default='Train',
                    help='path of train data')
parser.add_argument('--val-path', metavar='PATH', type=str, default='Test/Set14',
                    help='path to val data')
parser.add_argument('--test-path', metavar='PATH', type=str, default='Test/Set5',
                    help='path to test data')
parser.add_argument('-i', '--input_size', metavar='I', type=int, default=33,
                    help='size of input subimage for the model, the default\
                    value is aligned to the label size and the CNN\
                    architecture, make sure  you understand the network\
                    architecture if you want to change this value')
parser.add_argument('-l', '--label_size', metavar='L', type=int, default=21,
                    help='size of label subimage used to compute loss in CNN.\
                    The default value is aligned to the input and the CNN\
                    architecture, make sure you understand the network\
                    architecture if you want to change this value')
parser.add_argument('-s', '--stride', metavar='S', type=int, default=21,
                    help='This is not the stride in CNN, this is stride used\
                    for image subsampleing')
parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=32,
                    help='batch size used for training')
parser.add_argument('-n', '--num-epochs', metavar='N', type=int, default=50,
                    help='number of training epochs')
parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                    help='print training information')

args = parser.parse_args()

if args.phase not in ['train', 'test', 'both']:
    print('ERROR!!!')
    print('"Phase" must be "train" or "test"')
    print('')
    parser.print_help()
    sys.exit(1)

def main():
    print('############################################################')
    print('# SRCNN Pytorch implementation                             #')
    print('# by Thang Vu                                              #')
    print('############################################################')
    print('')
    print('-------YOUR SETTINGS_________')
    for arg in vars(args):
        print("%10s: %s" %(str(arg), str(getattr(args, arg))))
    print('')

    # config dataset
    common_config = {
        'scale': args.scale,
        'is_gray': True,
        'input_size': args.input_size,
        'label_size': args.label_size,
        'stride': args.stride
    }

    train_dataset_config = common_config.copy()
    val_dataset_config = common_config.copy()
    test_dataset_config = common_config.copy()

    train_dataset_config['dir_path'] = args.train_path
    val_dataset_config['dir_path'] = args.val_path
    test_dataset_config['dir_path'] = args.test_path

    print('Contructing dataset...')
    #construct dataset
    root = os.getcwd()
    root = os.path.join(root, 'overfit')

    if args.model == 'SRCNN_proposed':
        train_dataset = SR_dataset(root)
        val_dataset = SR_dataset(root)
        test_dataset = SR_dataset(root)
        model = SRCNN_proposed()
    elif args.model == 'SRCNN':
        train_dataset = SRCNN_dataset(root)
        val_dataset = SRCNN_dataset(root)
        test_dataset = SRCNN_datasaet(root)
        model = SRCNN()
    elif args.model == 'ESPCN':
        train_dataset = ESPCN_dataset(root)
        val_dataset = ESPCN_dataset(root)
        test_dataset = SRCNN_dataset(root)
        model = ESPCN()


    # use train_dataset val_dataset to train and validate the model
    datasets = {
        'train': train_dataset,
        'val': val_dataset
    }

    #model = SRCNN()
    solver = Solver(model, datasets, batch_size=args.batch_size,
                    num_epochs=args.num_epochs, verbose=args.verbose)

    if args.phase == 'train':
        print('Training...')
        solver.train()
    elif args.phase == 'test':
        print('Testing...')
        solver.test(test_dataset)
    else:
        print('Training...')
        solver.train()
        print('Testing...')
        solver.test(test_dataset)

if __name__ == '__main__':
    main()

