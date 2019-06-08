import torch
import torch.nn as nn


class my_vgg(nn.Module):
	def __init__(self, vgg):
		super(my_vgg, self).__init__()
		self.vgg = vgg
		self.avgpool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

	def forward(self, img):
		# x = img.unsqueeze(0)
		x1 = self.vgg.features[0](img)
		x2 = self.vgg.features[1](x1) # relu
		x3 = self.vgg.features[2](x2)
		x4 = self.vgg.features[3](x3) # relu
		x5 = self.avgpool(x4)

		x6 = self.vgg.features[5](x5)
		x7 = self.vgg.features[6](x6) # relu
		x8 = self.vgg.features[7](x7)
		x9 = self.vgg.features[8](x8) # relu
		x10 = self.avgpool(x9)

		x11 = self.vgg.features[10](x10)
		x12 = self.vgg.features[11](x11) # relu
		x13 = self.vgg.features[12](x12)
		x14 = self.vgg.features[13](x13) # relu
		x15 = self.vgg.features[14](x14)
		x16 = self.vgg.features[15](x15) # relu
		x17 = self.vgg.features[16](x16)
		x18 = self.vgg.features[17](x17) # relu
		x19 = self.avgpool(x18)

		x20 = self.vgg.features[19](x19)
		x21 = self.vgg.features[20](x20) # relu
		x22 = self.vgg.features[21](x21)
		x23 = self.vgg.features[22](x22) # relu
		x24 = self.vgg.features[23](x23)
		x25 = self.vgg.features[24](x24) # relu
		x26 = self.vgg.features[25](x25)
		x27 = self.vgg.features[26](x26) # relu
		x28 = self.avgpool(x27)

		x29 = self.vgg.features[28](x28)
		x30 = self.vgg.features[29](x29) # relu
		x31 = self.vgg.features[30](x30)
		x32 = self.vgg.features[31](x31) # relu
		x33 = self.vgg.features[32](x32)
		x34 = self.vgg.features[33](x33) # relu
		x35 = self.vgg.features[34](x34)
		x36 = self.vgg.features[35](x35) # relu

		return x4, x9, x18, x27, x36


class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input):
        return self.main(input)

class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        # 128 x 128
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64),
                vgg_layer(64, 64),
                )
        # 64 x 64
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                vgg_layer(128, 128),
                )
        # 32 x 32
        self.c3 = nn.Sequential(
                vgg_layer(128, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 256),
                )
        # 16 x 16
        self.c4 = nn.Sequential(
                vgg_layer(256, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        # 8 x 8
        self.c5 = nn.Sequential(
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512),
                )
        # 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(512, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.mp = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, input):
        h1 = self.c1(input) # 128 -> 64
        h2 = self.c2(self.mp(h1)) # 64 -> 32
        h3 = self.c3(self.mp(h2)) # 32 -> 16
        h4 = self.c4(self.mp(h3)) # 16 -> 8
        h5 = self.c5(self.mp(h4)) # 8 -> 4
        h6 = self.c6(self.mp(h5)) # 4 -> 1
        # return h6.view(-1, self.dim), [h1, h2, h3, h4, h5]
        return h6, [h1, h2, h3, h4, h5]


class decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder, self).__init__()
        self.dim = dim
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
                nn.ConvTranspose2d(dim, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # 8 x 8
        self.upc2 = nn.Sequential(
                vgg_layer(512*2, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 512)
                )
        # 16 x 16
        self.upc3 = nn.Sequential(
                vgg_layer(512*2, 512),
                vgg_layer(512, 512),
                vgg_layer(512, 256)
                )
        # 32 x 32
        self.upc4 = nn.Sequential(
                vgg_layer(256*2, 256),
                vgg_layer(256, 256),
                vgg_layer(256, 128)
                )
        # 64 x 64
        self.upc5 = nn.Sequential(
                vgg_layer(128*2, 128),
                vgg_layer(128, 64)
                )
        # 128 x 128
        self.upc6 = nn.Sequential(
                vgg_layer(64*2, 64),
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
                )
        self.up = nn.Upsample(scale_factor=2., mode='bilinear')

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec) # 1 -> 4
        # d1 = self.upc1(vec.view(-1, self.dim, 1, 1)) # 1 -> 4
        up1 = self.up(d1) # 4 -> 8
        d2 = self.upc2(torch.cat([up1, skip[4]], 1)) # 8 x 8
        up2 = self.up(d2) # 8 -> 16
        d3 = self.upc3(torch.cat([up2, skip[3]], 1)) # 16 x 16
        up3 = self.up(d3) # 16 -> 32
        d4 = self.upc4(torch.cat([up3, skip[2]], 1)) # 32 x 32
        up4 = self.up(d4) # 32 -> 64
        d5 = self.upc5(torch.cat([up4, skip[1]], 1)) # 64 x 64
        up5 = self.up(d5) # 64 -> 128
        output = self.upc6(torch.cat([up5, skip[0]], 1)) # 128 x 128
        return output


class Flow2Frame_warped(nn.Module):
    def __init__(self, num_channels):
        super(Flow2Frame_warped, self).__init__()
        # input shape [batch, 3, 128, 128]
        self.flow_encoder = encoder(dim=512, nc=2)
        self.image_encoder = encoder(dim=1024, nc=num_channels)
        self.image_decoder = decoder(dim=1024+512, nc=num_channels)

    def forward(self, warped_img, flow):
        img_hidden, img_skip = self.image_encoder(warped_img)
        flow_hidden, _ = self.flow_encoder(flow)
        concatenated_features = torch.cat(
            [img_hidden, flow_hidden], dim=1)
        # print concatenated_features.size()
        return self.image_decoder((concatenated_features, img_skip))


'''version without flow encoder'''
class RefineNet(nn.Module):
    def __init__(self, num_channels):
        super(RefineNet, self).__init__()
        # input shape [batch, 3, 128, 128]
        # self.flow_encoder = encoder(dim=512, nc=2)
        self.image_encoder = encoder(dim=1024, nc=num_channels)
        self.image_decoder = decoder(dim=1024, nc=num_channels)

    def forward(self, warped_img, flow):
        img_hidden, img_skip = self.image_encoder(warped_img)
        return self.image_decoder((img_hidden, img_skip))