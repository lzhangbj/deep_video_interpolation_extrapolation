import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np



class MotionDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm2d, num_D=3):
        super(MotionDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        kw = 3
        padw = int(np.ceil((kw-1.0)/2))
        # motion discriminator
        self.sequence = nn.Sequential(
                        nn.Conv2d(input_nc*3, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf), nn.LeakyReLU(),
                        # nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf),nn.LeakyReLU(),
                        nn.Conv2d(ndf, ndf, kernel_size=kw, stride=2, padding=padw), nn.BatchNorm2d(ndf),nn.LeakyReLU(),
                        nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf),nn.LeakyReLU(),
                        nn.Conv2d(ndf, 2*ndf, kernel_size=kw, stride=2, padding=padw), nn.BatchNorm2d(2*ndf),nn.LeakyReLU(),
                        nn.Conv2d(2*ndf, 2*ndf, kernel_size=3, stride=1, padding=1),

                        nn.BatchNorm2d(2*ndf),nn.LeakyReLU(), 
                        nn.Conv2d(2*ndf, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf),nn.LeakyReLU(), 
                        nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1) 
                        )
        # attention discriminator
        # self.sequence = nn.Sequential(
        #                 # nn.Conv2d(input_nc*3, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf), nn.LeakyReLU(),
        #                 # nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf),nn.LeakyReLU(),
        #                 nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.BatchNorm2d(ndf),nn.LeakyReLU(),
        #                 nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf),nn.LeakyReLU(),
        #                 nn.Conv2d(ndf, 2*ndf, kernel_size=kw, stride=2, padding=padw), nn.BatchNorm2d(2*ndf),nn.LeakyReLU(),
        #                 nn.Conv2d(2*ndf, 2*ndf, kernel_size=3, stride=1, padding=1)

        #                 # nn.BatchNorm2d(2*ndf),nn.LeakyReLU(), 
        #                 # nn.Conv2d(2*ndf, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf),nn.LeakyReLU(), 
        #                 # nn.Conv2d(ndf, 1, kernel_size=3, stride=1, padding=1) 
        #                 )      
        # self.seg_encoder = nn.Sequential(
        #                 nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
        #                 # nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(ndf),nn.LeakyReLU(),
        #                 nn.Conv2d(32, 32, kernel_size=kw, stride=1, padding=padw), nn.BatchNorm2d(32),nn.LeakyReLU(),
        #                 nn.Conv2d(32, 16, kernel_size=kw, stride=1, padding=padw), nn.BatchNorm2d(16),nn.LeakyReLU(),
        #                 nn.Conv2d(16, 8, kernel_size=kw, stride=1, padding=padw)
        #             )



    def singleD_forward(self, model, input):
        return [model(input)]

    def attention_mapv2(self, map1, map2, map3):
        bs,c,H,W = map1.size() 
        map1_normed = map1/torch.norm(map1, dim=1, keepdim=True)
        map2_normed = map2/torch.norm(map2, dim=1, keepdim=True)
        map3_normed = map3/torch.norm(map3, dim=1, keepdim=True)      

        # extract 3*3 patches,  map1_groups size (bs, c, H-2, W-2, 3, 3)
        patch_h = 1
        patch_w = 1
        map1_groups = map1_normed.unfold(dimension=2, size=patch_h, step=1).unfold(dimension=3, size=patch_w, step=1)\
                                .permute(0,2,3,1,4,5).view(bs, -1, c, patch_h, patch_w).contiguous()
        map3_groups = map3_normed.unfold(dimension=2, size=patch_h, step=1).unfold(dimension=3, size=patch_w, step=1)\
                                .permute(0,2,3,1,4,5).view(bs, -1, c, patch_h, patch_w).contiguous()

        forward_attmap = []
        forward_attmap_inds = []
        backward_attmap = []
        backward_attmap_inds = []
        for b in torch.arange(bs):
            # forward_singleb_attmap size (1, patch_h_num*patch_w_num, H, W)
            forward_singleb_attmap = F.conv2d(map2_normed[b].unsqueeze(0), map1_groups[b], stride=1, padding=0)
            forward_score_map, forward_inds_map = torch.max(forward_singleb_attmap, dim=1, keepdim=True)
            forward_attmap.append(forward_score_map) 
            forward_attmap_inds.append(forward_inds_map)

            backward_singleb_attmap = F.conv2d(map2_normed[b].unsqueeze(0), map3_groups[b], stride=1, padding=0)
            backward_score_map, backward_inds_map = torch.max(backward_singleb_attmap, dim=1,keepdim=True) 
            backward_attmap.append(backward_score_map) 
            backward_attmap_inds.append(backward_inds_map)

        forward_attmap = torch.cat(forward_attmap, dim=0)      
        backward_attmap = torch.cat(backward_attmap, dim=0)     
        attmap = (forward_attmap+backward_attmap)/2

        return attmap

    def attention_map(self, map1, map2, map3, h_range=5, w_range=9):
        bs,c,H,W = map1.size() 


        map1_normed = map1/torch.norm(map1, dim=1, keepdim=True)
        map2_normed = map2/torch.norm(map2, dim=1, keepdim=True)
        map3_normed = map3/torch.norm(map3, dim=1, keepdim=True)
        
        # calculate bidirectional attmap
        h_radius = h_range//2
        w_radius = w_range//2
        for_attmap= map1.new(bs, H, W, h_range, w_range).fill_(0)
        back_attmap = map1.new(bs, H, W, h_range, w_range).fill_(0)
        for h in torch.arange(h_range):
            for w in torch.arange(w_range):
                h_index = h-h_radius
                w_index = w-w_radius

                h_min = max(0, -h_index)
                h_max = min(H, H-h_index)
                w_min = max(0, -w_index)
                w_max = min(W, W-w_index)
                for_attmap[:, h_min:h_max, w_min:w_max, h, w] = \
                    torch.sum(map2_normed[:, :, h_min:h_max, w_min:w_max] \
                            * map1_normed[:, :, h_min+h_index:h_max+h_index, w_min+w_index:w_max+w_index], dim=1)
                back_attmap[:, h_min:h_max, w_min:w_max, h, w] = \
                    torch.sum(map2_normed[:, :, h_min:h_max, w_min:w_max] \
                            * map3_normed[:, :, h_min+h_index:h_max+h_index, w_min+w_index:w_max+w_index], dim=1)

        # get the correspondence index throught attmap
        for_attmap1d = torch.flatten(for_attmap, start_dim=3)
        back_attmap1d = torch.flatten(back_attmap, start_dim=3)
        for_attmap_max, for_attmap_index = torch.max(for_attmap1d, dim=3) # h*w + w
        back_attmap_max, back_attmap_index = torch.max(back_attmap1d, dim=3) # h*w + w

        score_map = (for_attmap_max + back_attmap_max) / 2
        score_map = score_map.view(bs, 1, H, W)

        # score_map[score_map<0.9] = 0
        # label_map = score_map.new(score_map.size()).fill_(1)
        # print("label_map", label_map.requires_grad)

        # label_map[score_map<0.9] = 0

        # index_map1d = for_attmap_index.new(bs, H*W)
        # for h in torch.arange(H):
        #     for w in torch.arange(W):
        #         index_map1d[:, h*W+w] = h*W + w

        # for_index_map1d = index_map1d + (torch.floor(for_attmap_index/w_range)-h_radius)*W + for_attmap_index%w_range



        # attention_map_index_mid_flattened_c = attention_map_index_mid_flattened.repeat(1, c, 1)

        # label_map = torch.gather(map1.new(bs, H*W).fill_(1),index=attention_map_index_mid_flattened, dim=1)

        # map1_flattened = map1.view(bs, c, H*W)
        # map2_flattened = map2.view(bs, c, H*W)
        # map2_gathered = torch.gather(map2_flattened, dim=2, index=attention_map_index_mid_flattened_c)    

        # corr_score = torch.sum(map1_flattened*map2_gathered, dim=1)
        # corr_map = corr_score[attention_map_index_mid_flattened] < 0.3
        
        # label_map[corr_map] = 0

        return score_map

    def forward(self, input, segs=None):
        # print("input", input.requires_grad)
        feat_maps = []
        result = []
        # for i in range(3):
        #     # seg = self.seg_encoder(segs[:, i*20:i*20+20])
        #     # feat_maps.append(self.sequence( torch.cat([seg, input[:,i*3:i*3+3]], dim=1)))
        #     feat_maps.append(self.sequence(input[:,i*3:i*3+3]))
        result.append( [F.sigmoid(self.sequence(input))])
        # result.append([self.attention_mapv2(*feat_maps)])
        return result
        