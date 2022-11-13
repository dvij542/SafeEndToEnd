import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import resnet as models
from torch.nn.utils.rnn import pack_padded_sequence
import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
from PIL import ImageFile
import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
SAFEGUARD = True
# image_size = 256
WIDTH = 336
HEIGHT = 180
RUN_NO = 6

TRAIN_STEERING = True
TRAIN_X_CURV_THETA = True
LOAD_PREVIOUS_MODEL = True


batch_size = 4
learning_rate = 1e-3
num_epochs = 10
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
        Y = image_path.split('.')[0].split('/')[1].split('_')
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
        return X.float(), Y1.float(), Y2.float()

def main(args):
    global RUN_NO, IMAGE_FOLDER, model_path, model_path_prev, LOAD_PREVIOUS_MODEL
    if args.run_no!=-1 :
        RUN_NO = args.run_no
    if SAFEGUARD :
        IMAGE_FOLDER = 'run'+str(RUN_NO)+'_cbf_images'
        model_path = 'saved_models_iter_cbf'+str(RUN_NO) 
        model_path_prev = 'saved_models_iter_cbf'+str(RUN_NO-1)
    else : 
        IMAGE_FOLDER = 'run'+str(RUN_NO)+'_images'
        model_path = 'saved_models_iter'+str(RUN_NO) 
        model_path_prev = 'saved_models_iter'+str(RUN_NO-1)
    if RUN_NO==0 :
        LOAD_PREVIOUS_MODEL = False
    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # print(IMAGE_FOLDER)
    image_paths = glob.glob(IMAGE_FOLDER+'/*.png')
    image_paths.sort()
    # print(image_paths)
    # Image preprocessing, normalization for the pretrained resnet
    cropped_dataset = croppedDataset(ims=image_paths,gt_output_loc=None)
    
    # Build data loader
    train_dl = DataLoader(cropped_dataset, batch_size, shuffle=True, num_workers=0)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    model = models.resnet18()
    in_features = model.fc.in_features
    model.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(model.fc.in_features, 1))
    model = model.cuda()
    # model.eval()
    
    model_safety_1 = models.resnet18()
    model_safety_1.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_safety_1.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_1 = model_safety_1.cuda()
    # print(model_safety_1)
    
    model_safety_2 = models.resnet18()
    model_safety_2.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_safety_2.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_2 = model_safety_2.cuda()
    print(model_safety_2)
    
    model_safety_3 = models.resnet18()
    model_safety_3.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model_safety_3.fc = nn.Sequential(nn.Dropout(0.2),nn.Linear(in_features, 1))
    model_safety_3 = model_safety_3.cuda()
    # print(model_safety_3)
    
    if LOAD_PREVIOUS_MODEL :
        st_dict = torch.load(os.path.join(model_path_prev, 'model-last.ckpt'))
        # st_dict['fc.1.weight'] = st_dict['fc.weight']
        # st_dict['fc.1.bias'] = st_dict['fc.bias']
        # del st_dict["fc.weight"]
        # del st_dict["fc.bias"]
        model.load_state_dict(st_dict)
        
        st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-1.ckpt'))
        # st_dict['fc.1.weight'] = st_dict['fc.weight']
        # st_dict['fc.1.bias'] = st_dict['fc.bias']
        # del st_dict["fc.weight"]
        # del st_dict["fc.bias"]
        model_safety_1.load_state_dict(st_dict)
        
        st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-2.ckpt'))
        # st_dict['fc.1.weight'] = st_dict['fc.weight']
        # st_dict['fc.1.bias'] = st_dict['fc.bias']
        # del st_dict["fc.weight"]
        # del st_dict["fc.bias"]
        model_safety_2.load_state_dict(st_dict)
        
        st_dict = torch.load(os.path.join(model_path_prev, 'model-last-safety-3.ckpt'))
        # st_dict['fc.1.weight'] = st_dict['fc.weight']
        # st_dict['fc.1.bias'] = st_dict['fc.bias']
        # del st_dict["fc.weight"]
        # del st_dict["fc.bias"]
        model_safety_3.load_state_dict(st_dict)
        
    params = list(model.parameters()) 
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    params_safety_1 = list(model_safety_1.parameters()) 
    optimizer_safety_1 = torch.optim.Adam(params_safety_1, lr=learning_rate)
    
    params_safety_2 = list(model_safety_2.parameters()) 
    optimizer_safety_2 = torch.optim.Adam(params_safety_2, lr=learning_rate)
    
    params_safety_3 = list(model_safety_3.parameters()) 
    optimizer_safety_3 = torch.optim.Adam(params_safety_3, lr=learning_rate)
    
    # exit(0)
    
    # Train the models
    total_step = len(train_dl)
    for epoch in tqdm.tqdm(range(num_epochs)):
        for i, (images, steerings, Y2) in enumerate(train_dl):
            # Set mini-batch dataset
            images = images.to(device)
            steerings = steerings.to(device)
            Y2 = Y2.to(device)
            
            # Forward, backward and optimize
            if TRAIN_STEERING :
                print(images.shape)
                outputs = model(images)
                loss = criterion(outputs, steerings)
                print(outputs,steerings)
                model.zero_grad()
                loss.backward()
                optimizer.step()
            
            if TRAIN_X_CURV_THETA :
                outputs_safety_1 = model_safety_1(images)
                # print(Y2[:,0:1])
                loss_safety_1 = criterion(outputs_safety_1, Y2[:,0:1])
                loss_safety_1_zero = criterion(outputs_safety_1*0, Y2[:,0:1])
                
                outputs_safety_2 = model_safety_2(images)
                # print(outputs_safety_2)#, Y2[:,1:2], Y2[:,3:4])
                # outputs_safety_2 = model_safety_2(images)
                # print(outputs_safety_2)#, Y2[:,1:2], Y2[:,3:4])
                # outputs_safety_2 = model_safety_2(images)
                # print(outputs_safety_2)#, Y2[:,1:2], Y2[:,3:4])
                loss_safety_2 = criterion(outputs_safety_2, Y2[:,1:2])
                loss_safety_2_zero = criterion(outputs_safety_2*0, Y2[:,1:2])
                
                outputs_safety_3 = model_safety_3(images)
                loss_safety_3 = criterion(outputs_safety_3, Y2[:,2:3])
                loss_safety_3_zero = criterion(outputs_safety_3*0, Y2[:,2:3])
                
                model_safety_1.zero_grad()
                model_safety_2.zero_grad()
                model_safety_3.zero_grad()
                
                loss_safety_1.backward()
                optimizer_safety_1.step()

                loss_safety_2.backward()
                optimizer_safety_2.step()

                loss_safety_3.backward()
                optimizer_safety_3.step()
            
            # Print log info
            if i % log_step == 0:
                if TRAIN_STEERING :
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item())) 
                    torch.save(model.state_dict(), os.path.join(
                        model_path, 'model-last.ckpt'))
                
                if TRAIN_X_CURV_THETA : 
                    print('Safety curvature : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss_safety_1.item())) 
                    torch.save(model_safety_1.state_dict(), os.path.join(
                        model_path, 'model-last-safety-1.ckpt'))
                    losses_1.append([loss_safety_1,loss_safety_1_zero])

                    print('Safety X : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss_safety_2.item())) 
                    torch.save(model_safety_2.state_dict(), os.path.join(
                        model_path, 'model-last-safety-2.ckpt'))
                    losses_2.append([loss_safety_2,loss_safety_2_zero])
                    
                    print('Safety theta : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch, num_epochs, i, total_step, loss_safety_3.item())) 
                    torch.save(model_safety_3.state_dict(), os.path.join(
                        model_path, 'model-last-safety-3.ckpt'))
                    losses_3.append([loss_safety_3,loss_safety_3_zero])
                    np.savetxt('losses_1.csv', np.array(losses_1))
                    np.savetxt('losses_2.csv', np.array(losses_2))
                    np.savetxt('losses_3.csv', np.array(losses_3))

                
                

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


