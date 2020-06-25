import cv2
import torch
import torchvision.transforms
import numpy as np
from utils import utils as utils
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='Video', help='which modality to train - Video\Audio\AV')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of lstm layers in the model')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='number of neurons in each lstm layer in the model')
    parser.add_argument('--debug', action='store_true', help='print debug outputs')
    args = parser.parse_args()

    # Load Model
    net = utils.import_network(args)
    net.load_state_dict(torch.load('/home/nas/user/kbh/End-to-End-VAD/saved_models/batch16/acc_76.314_epoch_040_arch_Video_state.pkl'))
    net.eval()

    # Opencv image params
    #cap = cv2.VideoCapture(0)   # 0: default camera
    cap = cv2.VideoCapture('/home/nas/user/kbh/End-to-End-VAD/data/Speaker8.avi')
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8

    # model params
    batchSize = 1
    timeDepth=15
    channels = 3
    size = 224

    # Opencv video params
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 30
    outputVid = cv2.VideoWriter('/home/nas/user/kbh/End-to-End-VAD/output.avi',fourcc,fps,(size,size))

    # Misc
    video_duration_in_frames = 3059

    #### Prep for Model ####

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    t = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                normalize,
            ])

    states= net.init_hidden(is_train=False)

    modelInput = torch.ones(batchSize,timeDepth,channels,size,size)
    idx=0

    print('START')
     
    while cap.isOpened():
        # 카메라 프레임 읽기
        success, frame = cap.read()

        if success:
            # resize
            frame = cv2.resize(frame, dsize=(size, size), interpolation=cv2.INTER_AREA)   
            # frame.shape : height , width, channel
            h = frame.shape[0]
            w = frame.shape[1]

            # channel, height, width
#            tensorFrame = np.transpose(frame,(2,0,1))
            tensorFrame = t(frame)

            # Convert to Tensor
#            tensorFrame = torch.cuda.FloatTensor(tensorFrame)
            tensorFrame = tensorFrame.to('cuda')
            #tensorFrame = tensorFrame.permute(2,0,1)

            if idx >=15:
                for i in range(14):
                    modelInput[0,i,:,:,:]=modelInput[0,i+1,:,:,:]

            i = min(idx,14)
            modelInput[0,i,:,:,:] = tensorFrame

            oupput=''
            idx=idx+1
            if idx >= 15 :
                output = net(modelInput.cuda(),states)
                output = output.to('cpu')

                output_1 = output.data[0][0].item()
                output_2 = output.data[0][1].item()

                outputString = '%f, %f'%(output_1,output_2)
                frame = cv2.putText(frame,outputString,(0,h),font,fontScale,(0,255,0),2)
                if output_2 > 0.9 : 
                    frame = cv2.putText(frame,'O',(10,20),font,fontScale,(0,0,255),2)
                else :
                    frame = cv2.putText(frame,'X',(10,20),font,fontScale,(0,0,255),2)
                print(str(idx)+' %f %f'%(output_1,output_2))

            outputVid.write(frame)

            #if idx >= video_duration_in_frames :
             #   break
        else :
            break;
    # Compose
           
    # Run Model
    cap.release()
    outputVid.release()

              



