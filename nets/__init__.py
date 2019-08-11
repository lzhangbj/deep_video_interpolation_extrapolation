from .MyFRRN import MyFRRN
from .GAN import GAN
from .VAE import VAE
from .UNet import UNet
from .SepUNet import SepUNet
from .VAE_S import VAE_S
from .B2SNet import B2SNet
from .SubNets import *
# from .SRN import SRN4, SRN4Seg, SRN4Sharp, HResUnet, AttnRefine, AttnBaseRefine, MSBaseRefine, \
# 				AttnRefineV2, AttnRefineV2Base, AttnRefineV2O, AttnRefineV3, AttnRefineV3Base, 
from .refine_nets import SRNRefine, MSResAttnRefine, MSResAttnRefineV2, MSResAttnRefineV2Base, MSResAttnRefineV3
from .RefineNet import RefineNet
from .RefineGAN import RefineGAN
from .multi_scale_discriminator import MultiscaleDiscriminator
from .PSPNet import PSPNet, PSPNetV2
from .HRNet import HRNet, InpaintUnet, VAEHRNet
from .ExtraNet import ExtraNet
from .ExtraInpaintNet import ExtraInpaintNet
from .InterNet import InterNet
from .InterRefineNet import InterRefineNet, InterStage3Net
from .InterGANNet import InterGANNet
from .FrameDisc import FrameDiscriminator, FrameLocalDiscriminator, FrameSNDiscriminator, FrameSNLocalDiscriminator
from .VidDisc import VideoDiscriminator, VideoLocalDiscriminator, VideoSNDiscriminator, VideoSNLocalDiscriminator
from .DetDisc import FrameDetDiscriminator, VideoDetDiscriminator, \
					FrameSNDetDiscriminator, VideoSNDetDiscriminator, \
					FrameLSSNDetDiscriminator, \
					VideoLSSNDetDiscriminator, VideoVecSNDetDiscriminator,VideoPoolSNDetDiscriminator, \
					VideoGlobalZeroSNDetDiscriminator, VideoGlobalResSNDetDiscriminator, \
					VideoGlobalMaskSNDetDiscriminator, VideoGlobalCoordSNDetDiscriminator
from .SpectralNorm import SpectralNorm
from .OpticalUnet import OpticalRefineNet, OpticalUnet, RefineUnet
from .TrackGen import TrackGen, TrackGenV2

