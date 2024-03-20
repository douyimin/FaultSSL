"""Copyright China University of Petroleum (East China), Yimin Dou, Kewen Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

# -*- coding:utf-8 -*-

import torch
import numpy as np
import argparse
import segyio
from utils import prediction, cubing_prediction, show_3DsliceView

parser = argparse.ArgumentParser(description='Prediction management')
parser.add_argument('--network', type=str, default=r'network/IOUFirst.pt',
                    help='Input cuboid path (.npy or .segy or .sgy)')
parser.add_argument('--input', type=str, default=r'data/F3.npy',
                    help='Input cuboid path (.npy or .segy or .sgy)')
parser.add_argument('--iline', type=int, default=189,
                    help='inline')  # 189 or 77, If none of them work, please fill the trace export in the commercial software and then read it.
parser.add_argument('--xline', type=int, default=193,
                    help='crossline')  # 193 or 73,If none of them work, please fill the trace export in the commercial software and then read it.
parser.add_argument('--infer_size', type=int, nargs='+', default=None,
                    # If you want to infer Kerry_full.npy, please use (768,176,176)
                    help='If None, the whole seismic volume is input.'
                         'If not None, the volume will be cut in blocks according to infer_size then input.'
                         'Shape = (tline,xline,iline) or (tline,iline,xline), must be divisible by 16')
parser.add_argument('--output_dir', type=str, default='output', help='Output dir')
parser.add_argument('--save_fault_cuboid', type=bool, default=False, help='')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # assert args.gamma == 0.5 or args.gamma == 0.6 or args.gamma == 0.7
    assert args.input[-3:] == 'npy' or args.input[-4:] == 'segy' or args.input[-3:] == 'sgy'
    model = torch.jit.load(args.network).to(device)

    if args.input[-3:] == 'npy':
        data = np.load(args.input)  # Shape = (tline,xline,iline) or (tline,iline,xline),
    else:
        data = segyio.open(args.input, iline=args.iline, xline=args.xline)
        data = segyio.cube(data)  # Shape = (tline,xline,iline) or (tline,iline,xline),
    if args.infer_size == None:
        infer_size = data.shape
    else:
        assert args.infer_size[0] >= data.shape[2], ("To ensure inference performance, it is essential to ensure that the tline length of  the cut blocks during inference is greater than or equal to the tline length of the seismic data."
                                                     f"For example, for this data, you can set the input size to ({data.shape[2]},{args.infer_size[1]},{args.infer_size[2]})")
        infer_size = (int(np.ceil(args.infer_size[0] / 16) * 16), int(np.ceil(args.infer_size[1] / 16) * 16),
                      int(np.ceil(args.infer_size[2] / 16) * 16))

    print('Load data successful.')
    print('Infer on', device)
    print(f'Data size is {tuple(data.shape)}, infer size is {infer_size}.')

    if args.infer_size == None:
        results = prediction(model, data)
    else:
        results = cubing_prediction(model, data, args.infer_size)

    show_3DsliceView(data, results, show_slice=(0.8, 0.025, 0.025))  # Ctrl+left click dragging profile.
    # show_slice: profile position.(tline,iline,xline)
