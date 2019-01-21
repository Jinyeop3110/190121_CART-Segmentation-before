from multiprocessing import Process
import os
import argparse
import torch
import torch.nn as nn
import sys
import utils
torch.backends.cudnn.benchmark = True

# example for mnist
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, '/home/Jinyeop/PyCharmProjects_JY/180801_2DcellSegmentation_JY')

#from slack_server import SlackBot
import datas.preprocess as preprocess

from Logger import Logger

from models.Fusionnet import Fusionnet
from models.unet import Unet2D
from datas.NucleusLoader import nucleusloader
from trainers.CNNTrainer import CNNTrainer
import copyreg

from loss import EdgeWeightedLoss, FocalLoss, TverskyLoss

"""parsing and configuration"""
def arg_parse():
    # projects description
    desc = "TBcell 2D segmentation"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--gpus', type=str, default="6,7",
                        help="Select GPU Numbering | 0,1,2,3 | ")
    parser.add_argument('--cpus', type=int, default="8",
                        help="Select CPU Number workers")
    parser.add_argument('--model', type=str, default='unet',
                        choices=['fusion', "unet", "unet_sh"], required=True)
    # Unet params
    parser.add_argument('--feature_scale', type=int, default=4)

    parser.add_argument('--in_channel', type=int, default=1)

    # FusionNet Parameters
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--clamp', type=tuple, default=None)

    parser.add_argument('--augment', type=str, default='',
                        help='The type of augmentaed ex) crop,rotate ..  | crop | flip | elastic | rotate |')

    # TODO : Weighted BCE
    parser.add_argument('--loss', type=str, default='l1',
                        choices=["l1", "l2"])

    #parser.add_argument('--data', type=str, default='data',
    #                    choices=['All', 'Balance', 'data', "Only_Label"],
    #                    help='The dataset | All | Balance | Only_Label |')

    parser.add_argument('--epoch', type=int, default=500, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--test', type=int, default=0, help='The size of batch')

    parser.add_argument('--save_dir', type=str, default='',
                        help='Directory name to save the model')

    # Adam Parameter
    parser.add_argument('--lrG', type=float, default=0.0005)
    parser.add_argument('--beta', nargs="*", type=float, default=(0.5, 0.999))

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

    copyreg.pickle(torch.dtype, pickle_torch_dtype)

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus
    torch_device = torch.device("cuda")
    #torch_device = torch.device("cpu")

    # manually change paths of training data and test data
    # filename example :
    f_path_train="/home/Jinyeop/PyCharmProjects_JY/180801_2DcellSegmentation_JY/data/Training_examples/total/train"
    f_path_test = "/home/Jinyeop/PyCharmProjects_JY/180801_2DcellSegmentation_JY/data/Training_examples/total/test"
    preprocess = preprocess.get_preprocess(arg.augment)

    train_loader = nucleusloader(f_path_train, arg.batch_size, transform=preprocess,
                                 cpus=arg.cpus,
                                 shuffle=True, drop_last=True)

    test_loader = nucleusloader(f_path_test, batch_size=1, transform=None,
                                 cpus=arg.cpus, shuffle=False,
                                drop_last=True)
    #test_loader = RBCLoader(f_path + "/test", batch_size=arg.batch_size,
    #                         transform=None,shuffle=True)


    if arg.model == "fusion":
        net = Fusionnet(arg.in_channel, arg.out_channel, arg.ngf, arg.clamp)
    elif arg.model == "unet":
        net = Unet2D(feature_scale=arg.feature_scale)
    elif arg.model == "unet_sh":
        net = UnetSH2D(arg.sh_size, feature_scale=arg.feature_scale)
    else:
        raise NotImplementedError("Not Implemented Model")

    net = nn.DataParallel(net).to(torch_device)
    if arg.loss == "l2":
        recon_loss = nn.L2Loss()
    elif arg.loss == "l1":
        recon_loss = nn.L1Loss()

    recon_loss=EdgeWeightedLoss(1,120)


    model = CNNTrainer(arg, net, torch_device, recon_loss=recon_loss, logger=logger)
    model.load(filename="epoch[0253]_losssum[225.786118].pth.tar")
    if arg.test==0:
        model.train(train_loader, test_loader)
    if arg.test==1:
        model.test(test_loader)
    # utils.slack_alarm("zsef123", "Model %s Done"%(arg.save_dir))

