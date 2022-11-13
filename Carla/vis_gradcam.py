from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import matplotlib.pyplot as plt


import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
from PIL import ImageFile
import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Device configuration
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# image_size = 256
WIDTH = 336
HEIGHT = 180
RUN_NO = 5

TRAIN_STEERING = True
TRAIN_X_CURV_THETA = True
LOAD_PREVIOUS_MODEL = True
IMAGE_FOLDER = 'example_imgs'

batch_size = 1
learning_rate = 1e-3
num_epochs = 20
save_step = 10
log_step = 1
transform = transforms.Compose([ 
        transforms.ToTensor(),
        # transforms.CenterCrop(min(HEIGHT,WIDTH)),
        # transforms.Resize(image_size),
        transforms.Normalize((0.485, 0.456, 0.406, 0.5), 
                             (0.229, 0.224, 0.225, 0.3))])

curvature_factor = 0.5
x_factor = 1
theta_factor = 0.1

losses_1 = []
losses_2 = []
losses_3 = []

class croppedDataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ims, gt_output_loc):
        'Initialization'
        self.ims = ims
        # self.gt_output = np.loadtxt(gt_output_loc,delimiter=',')
        self.transform = transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.ims)
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image_path = self.ims[index]
        # image_path = 'run4_images/frame_266_-5724_-127_1153_6373.png'
        # image_path = 'run3_images/frame_266_-5724_-127_1131_6321.png'
        image = Image.open(image_path)
        X = self.transform(image)
        # print(X.shape)
        X1 = X[:3,:HEIGHT,:]
        X2 = X[:3,HEIGHT:,:]
        X = torch.cat((X1,X2),0)
        # print(X.shape)
        Y = image_path.split('.')[0].split('_')[1:]
        # print(Y)
        cont_no = float(Y[1])
        steering = float(Y[2])/100.
        curvature = float(Y[5])/1000.
        x = float(Y[3])/100.
        theta = float(Y[4])/100.
        # print(Y)
        # print(curvature)
        Y1 = torch.tensor([steering])
        Y2 = torch.tensor([curvature/curvature_factor,x/x_factor,theta/theta_factor,cont_no])
        return X.float(), Y1.float(), Y2.float(), np.array(image)

def main(args):
    global RUN_NO, IMAGE_FOLDER, model_path, model_path_prev, LOAD_PREVIOUS_MODEL
    if args.run_no!=-1 :
        RUN_NO = args.run_no
    model_path = 'final_model_rc' 
    image_paths = glob.glob(IMAGE_FOLDER+'/*.png')
    image_paths.sort()
    cropped_dataset = croppedDataset(ims=image_paths,gt_output_loc=None)
    
    # Build data loader
    train_dl = DataLoader(cropped_dataset, batch_size, shuffle=True, num_workers=0)
    
    # Loss and optimizer
    model = models.resnet18()
    model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(model.fc.in_features, 1))
    print(model)
    # model = model.cuda()
    model.eval()
    
    if LOAD_PREVIOUS_MODEL :
        st_dict = torch.load(os.path.join(model_path, 'model-last.ckpt'))
        model.load_state_dict(st_dict)
        
    target_layers = [model.layer4[-1]]
    
    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers)#, use_cuda=args.use_cuda)

    for i, (images, steerings, Y2, image_orig) in enumerate(train_dl):
        # Set mini-batch dataset
        images = images.to(device)
        steerings = steerings.to(device)
        Y2 = Y2.to(device)
        
        # outputs = model(images)
        grayscale_cam = cam(input_tensor=images)

        # In this example grayscale_cam has only one image in the batch:
        # print(np.min(np.array(image_orig[0,:3,:HEIGHT,:])),np.max(np.array(image_orig[0,:3,:HEIGHT,:])))
        # print(grayscale_cam.shape,np.array(image_orig[0,:HEIGHT,:,:3]).shape)
        # grayscale_cam[:,:80]*=3
        visualization1 = show_cam_on_image(np.array(image_orig[0,:HEIGHT,:,:3])/255., grayscale_cam, use_rgb=True)
        visualization2 = show_cam_on_image(np.array(image_orig[0,HEIGHT:,:,:3])/255., grayscale_cam, use_rgb=True)
        print(i)
        plt.imsave('res'+str(i)+'_1.png',visualization1)
        plt.imsave('res'+str(i)+'_2.png',visualization2)
        
        
if __name__ == '__main__':
    # test()
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-r', '--run_no',
        metavar='P',
        default=-1,
        type=int,
        help='Run no')
    args = argparser.parse_args()
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    # parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    # parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    # parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    # parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    # parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    # parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # # Model parameters
    # parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    # parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    # parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    # parser.add_argument('--num_epochs', type=int, default=5)
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument('--learning_rate', type=float, default=0.001)
    # args = parser.parse_args()
    # print(args)
    # args = {}
    main(args)


