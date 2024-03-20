"""Copyright China University of Petroleum East China, Yimin Dou, Kewen Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import torch
import numpy as np
from torch.cuda import amp
from cigvis import colormap
import cigvis


def tensor_normalization(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / (_range + 1e-6)


def tensor_z_score_clip(data, clp_s=3.2):
    z = (data - torch.mean(data)) / torch.std(data)
    return tensor_normalization(torch.clip(z, min=-clp_s, max=clp_s))


def cubing_prediction(model, data, infer_size):
    model.eval()
    seis = data.transpose((2, 0, 1))
    tT, tH, tW = seis.shape
    infer_size1, infer_size2, infer_size3 = infer_size
    tTt = int(np.ceil(tT / infer_size1) * infer_size1)
    tHt = int(np.ceil(tH / infer_size2) * infer_size2)
    tWt = int(np.ceil(tW / infer_size3) * infer_size3)
    seis_tensor = torch.from_numpy(seis).float()[None, None]
    seis_tensor = torch.nn.ReflectionPad3d((0, tWt - tW, 0, tHt - tH, 0, tTt - tT))(seis_tensor)
    results = np.zeros_like(seis_tensor)[0, 0]
    for k in range(0, tTt, infer_size1):
        for i in range(0, tHt, infer_size2):
            for j in range(0, tWt, infer_size3):
                infer_cube = tensor_z_score_clip(
                    seis_tensor[:, :, k:k + infer_size1, i:i + infer_size2, j:j + infer_size3])
                with torch.no_grad():
                    with amp.autocast(True):
                        output_tmp = model(infer_cube.cuda()).cpu().numpy()[0, 0]
                results[k:k + infer_size1, i:i + infer_size2, j:j + infer_size3] = output_tmp
    results = results[:tT, :tH, :tW]
    return results.transpose((1, 2, 0))


def prediction(model, data):
    model.eval()
    seis = data.transpose((2, 0, 1))
    tT, tH, tW = seis.shape
    tTt = int(np.ceil(tT / 16) * 16)
    tHt = int(np.ceil(tH / 16) * 16)
    tWt = int(np.ceil(tW / 16) * 16)
    seis_tensor = torch.from_numpy(seis).float()[None, None]
    seis_tensor = tensor_z_score_clip(seis_tensor)
    seis_tensor = torch.nn.ReflectionPad3d((0, tWt - tW, 0, tHt - tH, 0, tTt - tT))(seis_tensor)
    with torch.no_grad():
        with amp.autocast(True):
            result = model(seis_tensor.cuda()).cpu().numpy()[0, 0, :tT, :tH, :tW]
    return result.transpose((1, 2, 0))


#
def show_3DsliceView(seis, fault, show_slice, likehood_show_threshold=0.5):
    fg_cmap = colormap.set_alpha_except_min('jet', alpha=1)
    fault = np.where(fault > likehood_show_threshold, fault, 0)
    iline, xline, tline = seis.shape

    nodes = cigvis.create_overlay(seis,
                                  fault,
                                  pos=[[int(iline * show_slice[1])], [int(xline * show_slice[2])],
                                       [int(tline * show_slice[0])]],
                                  bg_cmap='gray',
                                  fg_cmap=fg_cmap,
                                  fg_interpolation='nearest')
    cigvis.plot3D(nodes, size=(1500, 1500), savename='example.png')
