import torch
import argparse
import numpy as np
from utils import utils as utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   # parser.add_argument('--input', type=str, default='/home/nas/user/kbh/End-to-End-VAD/saved_models/acc_75.123_epoch_000_arch_Video_state.pkl', help='Torch state file to convert into Torchscript')
    parser.add_argument('--input', type=str, default='/home/kiosk/temp/End-to-End-VAD2/saved_models/Video/test/IIP_acc_92.205_epoch_002_arch_Video_edge_state.pkl', help='Torch state file to convert into Torchscript')
    parser.add_argument('--output', type=str, default='traced_state.pt', help='The name of output file')

    # Netwrok args
    parser.add_argument('--arch', type=str, default='Video', choices=['Video'], help='only Video')
    parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
    parser.add_argument('--lstm_layers', type=int, default=2, help='number of lstm layers in the model')
    parser.add_argument('--lstm_hidden_size', type=int, default=1024, help='number of neurons in each lstm layer in the model')
    parser.add_argument('--debug', action='store_true', help='print debug outputs')

    args = parser.parse_args()

    size = 224

    # only for Video model
    net = utils.import_network(args)

    # 
    trace_input=torch.rand(16,15,3,size,size).cuda()
    states = net.init_hidden(is_train=True)


    print('loading pre-trainded model..')
    net.load_state_dict(torch.load(args.input))

    print('eval')
    net.eval()

    print('tracing')
    # def forward(self, x, h):
    #     batch,frames,channels,height,width = x.squeeze().size()
    traced_model = torch.jit.trace(net, (trace_input,states))

    print('saving')
    traced_model.save(args.output)



