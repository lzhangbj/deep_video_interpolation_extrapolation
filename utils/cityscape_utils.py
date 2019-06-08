import numpy as np


seg_index2label={
	0:'unlabeled'            ,
    # 'ego vehicle'          (  0,  0,  0) ,
    # 'rectification border' (  0,  0,  0) ,
    # 'out of roi'           (  0,  0,  0) ,
    # 'static'               (  0,  0,  0) ,
    1:'dynamic'              ,
    2:'ground'                ,
    3:'road'                  ,
    4:'sidewalk'              ,
    5:'parking'             ,
    6:'rail track'            ,
    7:'building'              ,
    8:'wall'                 ,
    9:'fence'                ,
    10:'guard rail'           ,
    11:'bridge'               ,
    12:'tunnel'                ,

    13:'pole'                 ,	#
    # 'polegroup'            (153,153,153) ,
    14:'traffic light'         ,	#
    15:'traffic sign'         ,	#
    16:'vegetation'           ,	#

    17:'terrain'              ,
    18:'sky'                   ,

    19:'person'                ,	#
    20:'rider'                ,	#
    21:'car'                   ,	#
    22:'truck'                 ,	#
    23:'bus'                   ,	#
    24:'caravan'              ,	#
    25:'trailer'               ,	#
    26:'train'                 ,	#
    27:'motorcycle'            ,	#
    28:'bicycle'               
}

seg_id2index = {
-1:0,
0:0 ,       #unlabeled
1:0 ,       #ego vehicle
2:0 ,       #rectification border
3:0 ,       #out of roi
4:0 ,       #static
5:1 ,       #dynamic
6:2 ,       #ground
7:3 ,       #road   
8:4 ,       #sidewalk
9:5 ,       #parking
10:6 ,      #rail track
11:7 ,      #building
12:8 ,      #wall    
13:9 ,      #fence
14:10 ,     #guard rail
15:11 ,     #bridge
16:12 ,     #tunnel  
17:13 ,     #pole   
18:13 ,     #polegroup
19:14 ,     #traffic light
20:15 ,     #traffic sign
21:16 ,     #vegetation
22:17 ,     #terrain
23:18,      #sky    
24:19 ,     #person
25:20 ,     #rider
26:21 ,     #car   
27:22 ,     #truck
28:23 ,     #bus
29:24 ,     #caravan
30:25 ,     #trailer
31:26 ,     #train
32:27 ,     #motorcycle
33:28       #bicycle
}

seg_id2index_np = np.array(
    [0,
    0,
    0,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28])

# 30 colors -> 29 classes
# not used now
seg_color2index = {
	(  0,  0,  0 ):0,
	(  0,  0,142 ):0,
    (111, 74,  0):1,
    ( 81,  0, 81):2,
    (128, 64,128):3, 			#'road'
    (244, 35,232):4, 			#'sidewalk'            
    (250,170,160):5,			#'parking'              
    (230,150,140):6,			#rail track'            
    ( 70, 70, 70):7,			#building'              
    (102,102,156):8,			#wall'                  
    (190,153,153):9,			#fence'                 
    (180,165,180):10,			#guard rail'            
    (150,100,100):11,			#bridge'                
    (150,120, 90):12,			#tunnel'                

    (153,153,153):13,			#pole'                  
    (250,170, 30):14,			#traffic light'         
    (220,220,  0):15,			#traffic sign'          
    (107,142, 35):16,			#vegetation'            

    (152,251,152):17,			#terrain'              
    ( 70,130,180):18,			#sky'                   

    (220, 20, 60):19,			#person'                
    (255,  0,  0):20,			#rider'               
    (  0,  0,142):21,			#car'                   
    (  0,  0, 70):22,			#truck'                
    (  0, 60,100):23,			#bus'                   
    (  0,  0, 90):24,			#caravan'               
    (  0,  0,110):25,			#trailer'               
    (  0, 80,100):26,			#train'                
    (  0,  0,230):27,			#motorcycle'           
    (119, 11, 32):28			#bicycle'             	
}


colormap = np.zeros((20, 3), dtype=np.float32)
colormap[0] = [128, 64, 128]    # road
colormap[1] = [244, 35, 232]    # sidewalk
colormap[2] = [70, 70, 70]          # building
colormap[3] = [102, 102, 156]       # wall
colormap[4] = [190, 153, 153]       # fence
colormap[5] = [153, 153, 153]       # pole
colormap[6] = [250, 170, 30]        # traffic light
colormap[7] = [220, 220, 0]         # traffic sign
colormap[8] = [107, 142, 35]        # vegetation
colormap[9] = [152, 251, 152]       # terrain
colormap[10] = [70, 130, 180]       # sky
colormap[11] = [220, 20, 60]        # person
colormap[12] = [255, 0, 0]          # rider
colormap[13] = [0, 0, 142]          # car
colormap[14] = [0, 0, 70]           # truck
colormap[15] = [0, 60, 100]         # bus
colormap[16] = [0, 80, 100]         # on rails / train
colormap[17] = [0, 0, 230]          # motorcycle
colormap[18] = [119, 11, 32]        # bicycle


# img = Image.open("bochum_000000_009548_gtFine_myseg_id.png")

# img=np.array(img)
# print(img.shape)

# # print(img.shape)
# # print(img.dtype)
# # img = np.mean(img).astype(np.uint8)
# # print(np.min(img), np.max(img))
# img = colormap[img].astype(np.uint8)
# print(img.shape)
# # img = Image.fromarray(img,'RGB')
# print(img.size)
# scipy.misc.imsave("/data/linz/proj/file.png", img)
# # cv2.imwrite("/data/linz/file.png", img[:,:,::-1])
