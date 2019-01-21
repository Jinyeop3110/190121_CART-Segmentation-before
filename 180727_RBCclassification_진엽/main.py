from multiprocessing import Process
import os
import argparse
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True

# example for mnist
from torch.utils.data import Dataset, DataLoader
#from torchvision import datasets, transforms

#from slack_server import SlackBot
#import utils
import datas.preprocess as preprocess

from Logger import Logger

from models.FC_RBC_model import Net
from datas.RBCLoader import RBCLoader
from trainers.RBCTrainer import RBCTrainer
import copyreg

#from loss import FocalLoss, TverskyLoss

"""parsing and configuration"""
def arg_parse():
    # projects description
    desc = "MNIST Classifier"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="0,1,2,3",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="16",
                        help="Select CPU Number workers")
    parser.add_argument('--model', type=str, default='mnist',
                        choices=["FC"], required=True,
                        help='The type of Models | FC |')
    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    parser.add_argument('--augment', type=str, default='',
                        help='The type of augmentaed ex) crop,rotate ..  | crop | flip | elastic | rotate |')
    
    parser.add_argument('--epoch', type=int, default=300, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--infer', action="store_true", help='Only Inference')
    parser.add_argument('--test', action="store_true", help='Only Test')
    
    # Image meta data
    parser.add_argument('--in_channel', type=int, default='1',
                        help='The Channel of Input image')
    parser.add_argument('--out_channel', type=int, default='1',                        
                        help='The Channel of Output image')
    
    parser.add_argument('--loss', type=str, default='NLL',
                        choices=['BCE', "focal", "tversky", "MSE"],
                        help='The type of Loss Fuctions | BCE | focal | tversky | MSE |')
    parser.add_argument('--focal_gamma', type=float, default='2', help='gamma value of focal loss')
    parser.add_argument('--t_alpha', type=float, default='0.3', help='alpha value of tversky loss')

    parser.add_argument('--dtype', type=str, default='float',
                        choices=['float', 'half'],
                        help='The torch dtype | float | half |')


    parser.add_argument('--sampler', type=str, default='',
                        choices=['weight', ''],
                        help='The setting data sampler')
    
    # Adam Optimizer
    parser.add_argument('--lr_init',   type=float, default=0.0001)
    parser.add_argument('--beta',  nargs="*", type=float, default=(0.5, 0.999))
    
    return parser.parse_args()

def reconstruct_torch_dtype(torch_dtype: str):
    # a dtype string is "torch.some_dtype"
    dtype = torch_dtype.split('.')[1]
    return getattr(torch, dtype)

def pickle_torch_dtype(torch_dtype: torch.dtype):
    return reconstruct_torch_dtype, (str(torch_dtype),)

if __name__ == "__main__":
    arg = arg_parse()
    arg.save_dir = "%s/outs/%s"%(os.getcwd(), arg.save_dir)
    if os.path.exists(arg.save_dir) is False:
            os.mkdir(arg.save_dir)
    
    logger = Logger(arg.save_dir)
    #slack_bot = SlackBot(logger)
    #slack_proc = Process(target=slack_bot.run, name="Slack Process")
    #slack_proc.start()

    #os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cpu")
    copyreg.pickle(torch.dtype, pickle_torch_dtype)

    # manually change paths of training data and test data
    # filename example :
    f_path='C:/Users/SJY/PycharmProjects/180727_RBCclassification_진엽/data/Data_Normalized'
    preprocess = preprocess.get_preprocess(arg.augment)

    train_loader = RBCLoader(f_path + "/train", batch_size=arg.batch_size,
                             transform='smote',shuffle=True)
    test_loader = RBCLoader(f_path + "/test", batch_size=arg.batch_size,
                             transform=None,shuffle=True)

    ## Parameters

    arch_index = 5
    arg.l2_coeff = 0.0001
    #arg.repetition = 0
    #arg.dr = 0.995
    batch_size = 24
    epoch_size = 201
    arch = [[16], [16, 16], [16, 16, 16], [32], [32, 32], [32, 32, 32]]
    epoch_init = 0
    num_test = 40
    input_dim=int(train_loader.dataset.__getitem__(0)[0].size(0))
    print("input dimension : ",input_dim)
    output_dim=5
    print("output dimension : ",output_dim)

    if arg.model == "FC":
        net = Net('batchnorm{}'.format(arch_index), input_dim, output_dim, hidden_dims=arch[arch_index],
           use_batchnorm=True)
    #net = nn.DataParallel(net).to(torch_device)

    if arg.loss == "NLL":
        recon_loss = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError("Not Implemented Loss Function")
    
    model = RBCTrainer(arg, net, torch_device, recon_loss, logger)
    if arg.test is False:
        model.train(train_loader, test_loader)
    model.test(test_loader)
    slack_proc.terminate()
    # utils.slack_alarm("zsef123", "Model %s Done"%(arg.save_dir))

