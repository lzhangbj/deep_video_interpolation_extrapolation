import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np



class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)#.cuda()
            if getIntermFeat:                                
                for j in range(n_layers+1):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                # print(i, "input", result[-1].get_device())
                # try:
                #     print(i, 'model', model[i][0].weight.get_device())
                # except:
                #     print(i, 'sigmoid')
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+1)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers # 7

        kw = 3
        nf = ndf
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [    
                        [nn.Conv2d(input_nc, ndf, kernel_size=5, stride=1, padding=2), norm_layer(nf), nn.LeakyReLU(0.2, True)],
                        [nn.Conv2d(ndf, ndf, kernel_size=5, stride=1, padding=2), norm_layer(ndf), nn.LeakyReLU(0.2, True)],
                        [nn.Conv2d(ndf, 2*ndf, kernel_size=5, stride=2, padding=2), norm_layer(2*ndf), nn.LeakyReLU(0.2, True)],
                        [nn.Conv2d(2*ndf, 2*ndf, kernel_size=5, stride=1, padding=2), norm_layer(2*ndf), nn.LeakyReLU(0.2, True)],
                        [nn.Conv2d(2*ndf, 2*ndf, kernel_size=5, stride=2, padding=2), norm_layer(2*ndf), nn.LeakyReLU(0.2, True)],
                        [nn.Conv2d(2*ndf, 2*ndf, kernel_size=5, stride=1, padding=2), norm_layer(2*ndf), nn.LeakyReLU(0.2, True)],
                        [nn.Conv2d(2*ndf, 4*ndf, kernel_size=5, stride=4, padding=2), norm_layer(4*ndf), nn.LeakyReLU(0.2, True)],
                        [nn.Conv2d(4*ndf, 4*ndf, kernel_size=5, stride=1, padding=2), norm_layer(4*ndf), nn.LeakyReLU(0.2, True)],
                        [nn.Conv2d(4*ndf, 1, kernel_size=3, stride=1, padding=1)]
                        # [nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True)]
                    ]



        


        # sequence += [[nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        # base_layer = len(sequence)

        # nf = ndf
        # for n in range(base_layer, n_layers):
        #     nf_prev = nf
        #     nf = min(nf * 2, 512)
        #     sequence += [[
        #         nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
        #         norm_layer(nf), nn.LeakyReLU(0.2, True)
        #     ]]

        # nf_prev = nf
        # nf = min(nf * 2, 512)
        # sequence += [[
        #     nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
        #     norm_layer(nf),
        #     nn.LeakyReLU(0.2, True)
        # ]]

        # sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+1):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            assert torch.all( (res[-1]>=0) & (res[-1]<=1)), res[-1] 
            return res[1:]
        else:
            return self.model(input)   