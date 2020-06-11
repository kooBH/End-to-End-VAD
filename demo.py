import cv2
import torch
from utils import utils as utils
import argparse

def init_hidden(is_train):
    if is_train:
        return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
    else:
        return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='Video', help='which modality to train - Video\Audio\AV')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of lstm layers in the model')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='number of neurons in each lstm layer in the model')
    parser.add_argument('--debug', action='store_true', help='print debug outputs')
    args = parser.parse_args()

    # Load Model
    net = utils.import_network(args)
    net.load_state_dict(torch.load('/home/nas/user/kbh/End-to-End-VAD/saved_models/batch16/acc_75.123_epoch_000_arch_Video_state.pkl'))
    net.eval()
    states= net.init_hidden(is_train=False)

    #cap = cv2.VideoCapture(0)   # 0: default camera
    cap = cv2.VideoCapture("./data/Speaker1.avi") #동영상 파일에서 읽기

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3

    batchSize = 16
    timeDepth=15
    channels = 3
    size = 224

    video_duration_in_frames = 3059


    modelInput = torch.ones(batchSize,timeDepth,channels,size,size)
    idx=0
     
    while cap.isOpened():
        # 카메라 프레임 읽기
        success, frame = cap.read()

        if success:
            # resize
            frame = cv2.resize(frame, dsize=(size, size), interpolation=cv2.INTER_AREA)   

            # frame.shape : height , width, channel
            h = frame.shape[0]
            w = frame.shape[1]

            # Convert to Tensor

            tensorFrame = torch.cuda.FloatTensor(frame)
            tensorFrame = tensorFrame.permute(2,0,1)

            for i in range(max(idx,15)):
                modelInput[0,idx,:,:,:] = tensorFrame

            
            frame = cv2.putText(frame,"AA",(0,h),font,fontScale,(0,255,0),2)
            cv2.imshow('Camera Window', frame)

            # ESC를 누르면 종료
            key = cv2.waitKey(50) & 0xFF
            if (key == 27): 
                break

            idx=idx+1
            print(idx)

            if idx == 15 :
                break;

    # Compose

    # Run Model
    output = net(modelInput.cuda(),states)
    print(output)



    cap.release()
    cv2.destroyAllWindows()


