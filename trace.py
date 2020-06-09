import torch
import argparse
from utils import utils as utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='', help='Torch state file to convert into Torchscript')
    parser.add_argument('--output', type=str, default='traced_state.pt', help='The name of output file')
    args = parser.parse_args()

    if arg.input == '' :
        print("ERROR:: no input file ")
        exit(-1)

    # TODO
    trace_input= torch.rand(1, 3, 32, 32)

    # only for Video model
    net = utils.import_network('Video')

    print('loading pre-trainded model..')
    net.load_state_dict(torch.load(arg.input))

    print('eval')
    net.eval()

    print('tracing')
    traced_model = torch.jit.trace(net, trace_input)

    print('saving')
    traced_model.save(arg.output)



