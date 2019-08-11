import numpy as np
import os
import sys
import pickle
import json
import time
import glob
import shutil
# from scipy.misc import imread
import cv2
from PIL import Image
# from cfg import cfg

train_scenes = [
    "aachen",
    "bochum",
    "bremen",
    "cologne",
    "darmstadt",
    "dusseldorf",
    "erfurt",
    "hamburg",
    "hanover",
    "jena",
    "krefeld",
    "monchengladbach",
    "strasbourg",
    "stuttgart",
    "tubingen",
    "ulm",
    "weimar",
    "zurich"
]

val_scenes = [
    "frankfurt",	
    "lindau",		
    "munster"
]

test_scenes = [
    "berlin",  	 	
    "bielefeld", 	
    "bonn",			
    "leverkusen",	
    "mainz",		
    "munich"
]

scene_dict = {
	'train': train_scenes,
	'val': val_scenes,
	'test': test_scenes
}

clip_num_dict = {
	'train':2975,
	'val': 500,
	'test': 1525
}


def create_root_clip(load_dir, save_name,tail='leftImg8bit', ext='.png'):
    clip_dict = {}
    split_list = ['train', 'val', 'test']
    for split in split_list:
        scenes = scene_dict[split]
        clips = []
        for scene in scenes:
            file_temp = os.path.join(load_dir, split, scene, "*{}".format(tail+ext))
            files = glob.glob(file_temp)
            files = [file.split('/')[-1] for file in files]
            frames = [(int(t.split('_')[1]), int(t.split('_')[2])) for t in files]
            vid_idxes = set([t[0] for t in frames])
            for vid_idx in vid_idxes:
                frame_idxes = [frame[1] for frame in frames if frame[0]==vid_idx]
                frame_idxes.sort()
                cnt = 0
                while cnt < len(frame_idxes):
                    clip = []
                    for i in range(30):
                        frame_idx = frame_idxes[cnt]
                        frame_name = scene+'_'+str(vid_idx).zfill(6)+'_'+str(frame_idx).zfill(6)
                        frame_dir = os.path.join(split, scene, frame_name)
                        clip.append(frame_dir)
                        cnt += 1
                    clips.append(clip)
                    sys.stdout.write("\r<<<< {}/{} >>>".format(len(clips), clip_num_dict[split]))
        assert len(clips) == clip_num_dict[split], [len(clips), clip_num_dict[split]]
        clip_dict[split] = clips
    with open(save_name, 'wb') as f:
        pickle.dump(clip_dict, f)


def create_pred_clip(root_clip_file, save_name, interval=3, vid_len=3):
    with open(root_clip_file, 'rb') as f:
        root_clips = pickle.load(f)
    clip_choices = [range(t, 30, interval) for t in range(interval)]
    clip_idxes = []
    for choices in clip_choices:
        cnt = 0
        while cnt < len(choices):
            if cnt+vid_len <= len(choices):
                clip_idxes.append([choices[cnt+k] for k in range(vid_len)])
                cnt+=vid_len
                assert len(clip_idxes[-1]) == vid_len, [len(clip_idxes[-1]), vid_len] 
            else:
                break
    split_list=['train', 'val', 'test']
    clip_dict={}
    for split in split_list:
        clip_split = []
        for root_clip in root_clips[split]:
            for clip_idx in clip_idxes:
                clip_split.append([root_clip[ind] for ind in clip_idx])
            sys.stdout.write("\r<<<< {}/{} >>>".format(len(clip_split), clip_num_dict[split]))
        clip_dict[split] = clip_split
    with open(save_name, 'wb') as f:
        pickle.dump(clip_dict, f)

def create_interp_clip(root_clip_file, save_name, interval=3, vid_len=3):
    with open(root_clip_file, 'rb') as f:
        root_clips = pickle.load(f)
    clip_choices = [range(t, 30, interval) for t in range(interval)]
    clip_idxes = []
    for choices in clip_choices:
        cnt = 0
        while cnt < len(choices):
            if cnt+vid_len <= len(choices):
                clip_now = []
                for k in range(vid_len):
                    if k != (vid_len // 2):
                        clip_now.append(choices[cnt+k])
                clip_now.append(choices[cnt+ (vid_len//2)])
                clip_idxes.append(clip_now)
                cnt+=vid_len
            else:
                break
    split_list=['train', 'val', 'test']
    clip_dict={}
    for split in split_list:
        clip_split = []
        for root_clip in root_clips[split]:
            for clip_idx in clip_idxes:
                clip_split.append([root_clip[ind] for ind in clip_idx])
            sys.stdout.write("\r<<<< {}/{} >>>".format(len(clip_split), clip_num_dict[split]))
        clip_dict[split] = clip_split
    with open(save_name, 'wb') as f:
        pickle.dump(clip_dict, f)

def create_pred_lsclip(root_clip_file, save_name, interval=3, vid_len=3):
    with open(root_clip_file, 'rb') as f:
        root_clips = pickle.load(f)
    split_list=['train', 'val', 'test']
    clip_dict={}
    for split in split_list:
        clip_split = []
        for root_clip in root_clips[split]:
            for i in range(30):
                try:
                    clip_split.append([root_clip[k] for k in range(i, i+interval*(vid_len-1)+1, interval)])
                except:
                    break
            sys.stdout.write("\r<<<< {}/{} >>>".format(len(clip_split), clip_num_dict[split]))
        clip_dict[split] = clip_split
        print()
    with open(save_name, 'wb') as f:
        pickle.dump(clip_dict, f)


def create_interp_lsclip(root_clip_file, save_name, interval=3, vid_len=3):
    with open(root_clip_file, 'rb') as f:
        root_clips = pickle.load(f)
    split_list=['train', 'val', 'test']
    clip_dict={}
    for split in split_list:
        clip_split = []
        for root_clip in root_clips[split]:
            for i in range(30):
                try:
                    clip_split.append([root_clip[i], root_clip[i+interval*2], root_clip[i+interval]])
                except:
                    break
            sys.stdout.write("\r<<<< {}/{} >>>".format(len(clip_split), clip_num_dict[split]))
        clip_dict[split] = clip_split
    with open(save_name, 'wb') as f:
        pickle.dump(clip_dict, f)

# create_pred_lsclip("/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl",
#             "/data/linz/proj/Dataset/Cityscape/load_files/int_{}_len_{}_extra_lsclip.pkl".format(5, 3), 5, 3)

def create_bbox_file(clip_file, bbox_src_dir, save_name, interval=9, vid_len=3):
    with open(clip_file,'rb') as f:
        clips_all = pickle.load(f)
    both_have_list = []
    only_for_list  = []
    only_back_list = []
    none_have_list = []
    eliminate_thresh = 0.9
    gen_dict = {'val':[]}

    with open(save_name, 'rb') as f:
        load_clip = pickle.load(f)
        gen_dict.update(load_clip)

    for mode in ['val']:
        clips = clips_all[mode]
        for clip_ind, clip in enumerate(clips):
            search_dir = os.path.join(bbox_src_dir, clip[1])
            bbox_files = glob.glob(search_dir+'/*.txt')
            bbox_files.sort()
            assert len(bbox_files) == 2*interval+1, len(bbox_files)
            mid_frame_file = bbox_files[interval]
            mid_objs = read_bbox_file(mid_frame_file)
            if mid_objs is None:
                gen_dict[mode].append([[],[],[]])
                only_for_list.append(0)
                only_back_list.append(0)
                both_have_list.append(0)
                none_have_list.append(0)
                continue
            src_obj_num = len(mid_objs)
            record_mid_objs = [obj[1:] for obj in mid_objs]
            objs_record = [[0]*src_obj_num, record_mid_objs, [0]*src_obj_num]
            for_frames = bbox_files[:interval][::-1]
            for i,for_frame in enumerate(for_frames):
                for_frame_objs = read_bbox_file(for_frame)
                assert len(for_frame_objs) == src_obj_num
                for t in range(src_obj_num):
                    if objs_record[0][t] is not None:
                        objs_record[0][t] = None if for_frame_objs[t][0]<eliminate_thresh else for_frame_objs[t][1:] 

            back_frames = bbox_files[interval+1:]
            for i,back_frame in enumerate(back_frames):
                back_frame_objs = read_bbox_file(back_frame)
                assert len(back_frame_objs) == src_obj_num
                for t in range(src_obj_num):
                    if objs_record[2][t] is not None:
                        objs_record[2][t]=None if back_frame_objs[t][0]<eliminate_thresh else back_frame_objs[t][1:] 
            gen_dict[mode].append(objs_record)
            # record statistics
            have_for = [obj is not None for obj in objs_record[0]] 
            have_back = [obj is not None for obj in objs_record[2]] 
            have_both = [ have_for[i] and have_back[i] for i in range(src_obj_num) ]
            have_only_for = [ have_for[i] and not have_back[i] for i in range(src_obj_num) ]
            have_only_back = [ not have_for[i] and have_back[i] for i in range(src_obj_num) ]
            both_have_list.append(have_both.count(True))
            only_for_list.append(have_only_for.count(True))
            only_back_list.append(have_only_back.count(True))
            none_have_list.append(src_obj_num-both_have_list[-1]-only_for_list[-1]-only_back_list[-1])
            sys.stdout.write('\r {}/{}'.format(clip_ind+1, len(clips)))
        print()

        both_have_avg = np.average(both_have_list)
        both_have_std = np.std(both_have_list)

        only_for_avg = np.average(only_for_list)
        only_for_std = np.std(only_for_list)

        only_back_avg = np.average(only_back_list)
        only_back_std = np.std(only_back_list)

        none_have_avg = np.average(none_have_list)
        none_have_std = np.std(none_have_list)
        print(mode)
        print('both_have avg={} std={}'.format(both_have_avg,both_have_std))
        print('only_for  avg={} std={}'.format(only_for_avg,only_for_std))
        print('only_back avg={} std={}'.format(only_back_avg,only_back_std))
        print('none_have avg={} std={}'.format(none_have_avg,none_have_std))

    with open(save_name, 'wb') as f:
        pickle.dump(gen_dict, f)

def read_bbox_file(file):
    with open(file) as f:
        objs=f.readline().strip('\n')
        if len(objs) == 0: # no object detected
            return None
        obj_list = objs.split('---')
        obj_list = [obj.split(',') for obj in obj_list]
        obj_list = [[float(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])] for obj in obj_list]
    return obj_list            

# create_bbox_file(   '/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_extra_lsclip.pkl',
#                     '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_panet_i9_track', 
#                     '/data/linz/proj/Dataset/Cityscape/obj_coords/int_9_len_3_extra_512x1024_panet_lsclip.pkl', 9, 3)

def clean_bbox_file(src_bbox_file, src_clip_file, save_bbox_file, save_clip_file):
    with open(src_bbox_file, 'rb') as f:
        clips = pickle.load(f)
        clips = clips['train']
    with open(src_clip_file, 'rb') as f:
        clip_clips = pickle.load(f)
        clip_clips = clip_clips['train']
    NUM_DET_BOX   = 10
    NUM_TRACK_BOX = 10
    gen_dict = {'train':[]}
    gen_clip_dict = {'train':[]}
    for tt,clip in enumerate(clips):
        if len(clip[1]) < NUM_DET_BOX:
            continue
        new_clip_0 = []
        new_clip_1 = []
        new_clip_2 = []
        valid_mid_clip_check = [t is not None for t in clip[1]]
        for ii, i in enumerate(valid_mid_clip_check):
            if i:
                new_clip_1.append(clip[1][ii])
                new_clip_0.append(clip[0][ii])
                new_clip_2.append(clip[2][ii])
        clip[1] = new_clip_1
        clip[0] = new_clip_0
        clip[2] = new_clip_2
        if len(clip[1]) < NUM_DET_BOX:
            continue
        forward_clip_check = [t is not None for t in clip[0]]
        backward_clip_check = [t is not None for t in clip[2]]
        both_clip_check = [forward_clip_check[i] and backward_clip_check[i] for i in range(len(clip[1]))]
        if both_clip_check.count(True) < NUM_TRACK_BOX:
            continue
        record_box = [[],[],[]]
        valid_inds = []
        for i in range(len(both_clip_check)):
            if both_clip_check[i]:
                valid_inds.append(i)
        valid_inds = valid_inds[:NUM_TRACK_BOX]
        for ind in valid_inds:
            record_box[0].append(clip[0][ind])
            record_box[1].append(clip[1][ind])
            record_box[2].append(clip[2][ind])
        for box in record_box[1]:
            assert box is not None, record_box
        gen_dict['train'].append(record_box)
        gen_clip_dict['train'].append(clip_clips[tt])

    print(len(gen_dict['train']))
    with open(save_bbox_file, 'wb') as f:
        pickle.dump(gen_dict, f)
    with open(save_clip_file, 'wb') as f:
        pickle.dump(gen_clip_dict, f)

# clean_bbox_file('/data/linz/proj/Dataset/Cityscape/obj_coords/int_9_len_3_extra_512x1024_panet_lsclip.pkl',
#                 '/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_extra_lsclip.pkl',
#                 '/data/linz/proj/Dataset/Cityscape/obj_coords/int_9_len_3_extra_512x1024_10bb_panet_lsclip.pkl',
#                 '/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_10bb_extra_panet_lsclip.pkl')

def area_check(bbox, thresh=6000):
    return float((bbox[2]-bbox[0]) * (bbox[3] - bbox[1])) > thresh

def area_ratio(bbox, h, w):
    '''
    '''
    return float((bbox[2]-bbox[0]) * (bbox[3] - bbox[1]))/(h*w)

def clean_bbox_file_max(src_bbox_file, src_clip_file, save_bbox_file, save_clip_file, num_box=8, area_thresh=0):
    with open(src_bbox_file, 'rb') as f:
        clip_all = pickle.load(f)
        # clips = clips['val']
    with open(src_clip_file, 'rb') as f:
        clip_clip_all = pickle.load(f)
        # clip_clip = clip_clips['val']
    MAX_NUM_DET_BOX   = num_box
    MAX_NUM_TRACK_BOX = num_box
    gen_dict = {'train':[],'val':[]}
    gen_clip_dict = {'train':[], 'val':[]}
    obj_sum=0
    obj_cnt=0
    # with open(save_bbox_file, 'rb') as f:
    #     load_bbox_file = pickle.load(f)
    #     gen_dict.update(load_bbox_file)
    # with open(save_clip_file, 'rb') as f:
    #     load_bbox_file = pickle.load(f)
    #     gen_clip_dict.update(load_bbox_file)

    for mode in ['train','val']:
        clips = clip_all[mode]
        clip_clips = clip_clip_all[mode]
        for tt,clip in enumerate(clips):
            if len(clip[1]) == 0:
                continue
            new_clip_0 = []
            new_clip_1 = []
            new_clip_2 = []
            valid_mid_clip_check = [t is not None and area_check(t, area_thresh) for t in clip[1]]
            for ii, i in enumerate(valid_mid_clip_check):
                if i:
                    new_clip_1.append(clip[1][ii])
                    new_clip_0.append(clip[0][ii])
                    new_clip_2.append(clip[2][ii])
            clip[1] = new_clip_1
            clip[0] = new_clip_0
            clip[2] = new_clip_2
            if len(clip[1]) == 0:
                continue
            forward_clip_check = [t is not None  for t in clip[0]]
            backward_clip_check = [t is not None for t in clip[2]]
            both_clip_check = [forward_clip_check[i] and backward_clip_check[i] for i in range(len(clip[1]))]
            if both_clip_check.count(True) == 0:
                continue
            record_box = [[],[],[]]
            valid_inds = []
            for i in range(len(both_clip_check)):
                if both_clip_check[i]:
                    valid_inds.append(i)
            valid_inds = valid_inds[:MAX_NUM_TRACK_BOX]
            track_num =  len(valid_inds)
            for ind in valid_inds:
                record_box[0].append(clip[0][ind])
                record_box[1].append(clip[1][ind])
                record_box[2].append(clip[2][ind])
            obj_sum+=track_num
            obj_cnt+=1
            rand_augment_index = 0
            while len(record_box[1]) <  MAX_NUM_TRACK_BOX:
                rand_augment_index = (rand_augment_index+1)%track_num
                record_box[0].append(record_box[0][rand_augment_index].copy())
                record_box[1].append(record_box[1][rand_augment_index].copy())
                record_box[2].append(record_box[2][rand_augment_index].copy())
            for box in record_box[1]:
                assert box is not None, record_box
            for seq_ind in range(3):
                for box_ind in range(len(record_box[seq_ind])):
                    area_rat = area_ratio(record_box[seq_ind][box_ind], 512, 1024)
                    record_box[seq_ind][box_ind].insert(0, area_rat) 
            gen_dict[mode].append(record_box)
            gen_clip_dict[mode].append(clip_clips[tt])
        print(mode)
        print(len(gen_dict[mode]))
        print(obj_sum/float(obj_cnt))
    with open(save_bbox_file, 'wb') as f:
        pickle.dump(gen_dict, f)
    with open(save_clip_file, 'wb') as f:
        pickle.dump(gen_clip_dict, f)

clean_bbox_file_max('/data/linz/proj/Dataset/Cityscape/obj_coords/int_9_len_3_extra_512x1024_panet_lsclip.pkl',
                '/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_extra_lsclip.pkl',
                '/data/linz/proj/Dataset/Cityscape/obj_coords/int_9_len_3_extra_512x1024_max_10bb_area_3000_panet_lsclip_check.pkl',
                '/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_max_10bb_area_3000_extra_panet_lsclip_check.pkl', 
                num_box=10, area_thresh=3000)

def combine_pkl_files(src_bbox_file_1, src_clip_file_1, src_bbox_file_2, src_clip_file_2, save_bbox_file, save_clip_file):
    with open(src_bbox_file_1, 'rb') as f:
        bboxes_1 = pickle.load(f)
        bboxes_1 = bboxes_1['train']
    with open(src_clip_file_1, 'rb') as f:
        clips_1 = pickle.load(f)
        clips_1 = clips_1['train']   
    with open(src_bbox_file_2, 'rb') as f:
        bboxes_2 = pickle.load(f)
        bboxes_2 = bboxes_2['train']
    with open(src_clip_file_2, 'rb') as f:
        clips_2 = pickle.load(f)
        clips_2 = clips_2['train']   

    new_bboxes = {'train':bboxes_1+bboxes_2, 
                    'val':[]}
    with open(save_bbox_file, 'wb') as f:
        pickle.dump(new_bboxes, f)

    new_clips = {'train':clips_1+clips_2, 
                    'val':[]}
    with open(save_clip_file, 'wb') as f:
        pickle.dump(new_clips, f)

# combine_pkl_files('/data/linz/proj/Dataset/Cityscape/obj_coords/int_9_len_3_extra_512x1024_max_10bb_area_3000_panet_lsclip.pkl',
#                  '/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_max_10bb_area_3000_extra_panet_sclip.pkl',
#                  '/data/linz/proj/Dataset/Cityscape/obj_coords/int_5_len_3_extra_512x1024_max_10bb_area_3000_panet_lsclip.pkl',
#                  '/data/linz/proj/Dataset/Cityscape/load_files/int_5_len_3_max_10bb_area_3000_extra_panet_sclip.pkl',

#                  '/data/linz/proj/Dataset/Cityscape/obj_coords/int_95_len_3_extra_512x1024_max_10bb_area_3000_panet_lsclip.pkl',
#                  '/data/linz/proj/Dataset/Cityscape/load_files/int_95_len_3_max_10bb_area_3000_extra_panet_sclip.pkl',
#                  )

def temp_move(src_dir, new_dir):
    txt_files = glob.glob(src_dir+'/*/*/*/*.txt', recursive=True)
    for t, txt_file in enumerate(txt_files):
        name = '/'.join(txt_file.split('/')[-4:])
        new_name = os.path.join(new_dir, name)
        new_file_dir = '/'.join(new_name.split('/')[:-1])
        if not os.path.exists(new_file_dir):
            os.makedirs(new_file_dir)
        # print(txt_file, new_name)
        shutil.copyfile(txt_file, new_name)
        sys.stdout.write('\r {}/{}'.format(t+1, len(txt_files)))

# temp_move('/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_i5_track',
#             '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_i5_track_txt')

def clean_panet_bboxes():
    src_dir = '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_panet_det'
    dest_dir = '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_panet_cleaned_corner_det'
    files = glob.glob(src_dir+'/val/*/*.txt')
    print(len(files))
    cnt_all = 0
    cls_record={}
    tic = time.time()
    for ind, file in enumerate(files):
        with open(file, 'r') as f:
            strs = f.readline().strip()
            bbox_list = strs.split('---')
            bbox_rec_list = [bbox.split(',') for bbox in bbox_list]
        new_bbox_list = []
        if bbox_rec_list == [['']]:
            new_bbox_list = ['']
        else:
            for bbox_desc in bbox_rec_list:
                assert len(bbox_desc) == 6, bbox_rec_list
                cls_name = bbox_desc[0]
                score = float(bbox_desc[1])
                assert score >= 0.6 and score <= 1, score
                valid = valid_panet_box(bbox_desc)
                if not valid:
                    print(bbox_desc)
                if valid:
                    bbox_desc = trans_bbox_to_corner(bbox_desc)
                    if bbox_desc[0] in cls_record:
                        cls_record[bbox_desc[0]] += 1
                    else:
                        cls_record[bbox_desc[0]] = 1

                    bbox_str = ','.join(bbox_desc)
                    new_bbox_list.append(bbox_str)
        cnt_all+=len(new_bbox_list)
        new_bbox_str = '---'.join(new_bbox_list)
        save_dir = file.replace(src_dir, dest_dir)
        prefix = '/'.join(save_dir.split('/')[:-1])
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        with open(save_dir, 'w') as f:
            f.write(new_bbox_str)
        if ind%100 == 0:
            tac = time.time()
            cost = tac-tic
            tic=time.time()
            sys.stdout.write('\r {}/{} time: {:.1f}s  obj: {:d}'.format(ind+1, len(files), cost, int(float(cnt_all)/(ind+1))))
    print()
    for key,value in cls_record.items():
        print(key, value)


def valid_panet_box(bbox_desc):
    x1, y1, w, h = tuple(map(int, bbox_desc[2:]))
    x2 = x1+w
    y2 = y1+h
    valid = (    (x1>=0 and x1<1024)\
            and (x2>=0 and x2<1024)\
            and (y1>=0 and y1<512)\
            and (y2>=0 and y2<512)\
            and (x1<x2 and y1<y2)   )
    return valid


def trans_bbox_to_corner(bbox_desc):
    x1, y1, w, h = tuple(map(int, bbox_desc[2:]))
    x2 = x1+w
    y2 = y1+h
    bbox_desc[4] = str(x2)
    bbox_desc[5] = str(y2) 
    return bbox_desc

# clean_panet_bboxes()


def create_kitti_format_cityscape(clip_file, ori_dir, new_dir):
    with open(clip_file, 'rb') as f:
        clips = pickle.load(f)
        clips = clips['val']

    vid_len = len(clips[0])
    mid_ind = vid_len//2
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for ind, clip in enumerate(clips):
        ind += 2975
        clip_dir = os.path.join(new_dir, "{:0>4d}".format(ind))
        if not os.path.exists(clip_dir):
            os.makedirs(clip_dir)
        ori_clip_names = [ori_dir+'/'+frame+"_leftImg8bit.png" for frame in clip]
        new_clip_names = [clip_dir+'/'+frame.split('/')[-1]+'_leftImg8bit.png' for frame in clip]
        assert len(clip) == 30, len(clip)
        for i in range(30):
            shutil.copyfile(ori_clip_names[i], new_clip_names[i])
        sys.stdout.write('\r {}/{} '.format(ind+1, len(clips)))
    print()

# create_kitti_format_cityscape(  '/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl',
#                                 '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024',
#                                 '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_kitti_format/images'
#                                 )

def set_in_int_range(data, min_val, max_val):
    if data < min_val:
        data=min_val
    if data > max_val:
        data=max_val
    data = int(data)
    return data


def clean_trackrcnn_tracking_data(clip_pkl, data_dir, new_data_dir):
    with open(clip_pkl, 'rb') as f:
        clips = pickle.load(f)
    txt_files = glob.glob(data_dir+'/*.txt')
    for id, file in enumerate(txt_files):
        if id>=2975:
            mode='val'
            true_id=id-2975
        else:
            mode='train'
            true_id=id

        clip = clips[mode][true_id]
        # make prefix dir
        clip_sub_dir = '/'.join(clip[0].split('/')[:-1])
        clip_dir = os.path.join(new_data_dir, clip_sub_dir)

        if not os.path.exists(clip_dir):
            os.makedirs(clip_dir)
        with open(file, 'r') as load_f:
            lines = load_f.readlines()
            save_files = [open(os.path.join(new_data_dir, clip[frame_id]+ '_leftImg8bit.txt') , 'w') for frame_id in range(30)]
            for line in lines:
                line = line.strip()
                line_split = line.split(' ')
                # frame_id, obj_id, x1, x2, y1, y2, score
                frame_id = int(line_split[0])
                assert frame_id >= 0 and frame_id <= 29, frame_id
                useful_line = line_split[1:2] + line_split[5:10]
                useful_line = list(map(float, useful_line))
                useful_line[0] = int(useful_line[0])
                useful_line[1] = set_in_int_range(useful_line[1], 0, 1023)
                useful_line[2] = set_in_int_range(useful_line[2], 0, 511)
                useful_line[3] = set_in_int_range(useful_line[3], 0, 1023)
                useful_line[4] = set_in_int_range(useful_line[4], 0, 511)
                useful_line = ','.join(list(map(str, useful_line))) + '\n'
                save_files[frame_id].write(useful_line)
            for save_file in save_files:
                save_file.close()
        sys.stdout.write('\r {}/{}'.format(id+1, len(txt_files)))

# clean_trackrcnn_tracking_data(  '/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl',
#                                 '/data/linz/proj/Dataset/TrackR-CNN-master/forwarded/conv3d_sep2/tracking_data',
#                                 '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_trackrcnn_root_track')

def create_trackrcnn_track_data(clip_pkl, root_data_dir, new_data_dir):
    with open(clip_pkl, 'rb') as f:
        clips_all = pickle.load(f)

    def read_in_file(file):
        with open(file, 'r') as f:
            objs = f.readlines()
            objs = [obj.strip() for obj in objs]
            objs = [obj.split(',') for obj in objs]
            obj_dict = {}
            for obj in objs:
                obj_id = int(obj[0])
                assert obj_id not in obj_dict, 'repetition among objs !'
                obj_content = [float(obj[-1])] + list(map(int, obj[-5:-1]))
                obj_dict[obj_id] = obj_content
        return obj_dict

    for mode in ['train', 'val']:
        clips = clips_all[mode]
        for clip_ind, clip in enumerate(clips):
            save_dir = os.path.join(new_data_dir, clip[1])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_files = [os.path.join(save_dir, clip[i].split('/')[-1]+'_leftImg8bit.txt') for i in range(3)]
            save_files = [open(save_file, 'w') for save_file in save_files]
            mid_objs = read_in_file(os.path.join(root_data_dir, clip[1]+'_leftImg8bit.txt'))
            for_objs = read_in_file(os.path.join(root_data_dir, clip[0]+'_leftImg8bit.txt'))
            back_objs = read_in_file(os.path.join(root_data_dir, clip[2]+'_leftImg8bit.txt'))
            write_al = False
            for obj_id in mid_objs:
                if obj_id in for_objs and obj_id in back_objs:
                    for_obj  = list(map(str, for_objs[obj_id]))
                    for_line = ','.join(for_obj)
                    mid_obj  = list(map(str, mid_objs[obj_id]))
                    mid_line = ','.join(mid_obj)
                    back_obj = list(map(str, back_objs[obj_id]))
                    back_line = ','.join(back_obj)
                    if write_al:
                        save_files[0].write('---')
                        save_files[1].write('---')
                        save_files[2].write('---')
                    else:
                        write_al=True
                    save_files[0].write(for_line)
                    save_files[1].write(mid_line)
                    save_files[2].write(back_line)
            for save_file in save_files:
                save_file.close()
            sys.stdout.write('\r {}/{} {}'.format(clip_ind+1, len(clips), mode))
        print()

# create_trackrcnn_track_data('/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_extra_lsclip.pkl',
#                             '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_trackrcnn_root_track',
#                             '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_trackrcnn_i9_track')

def create_trackrcnn_track_pkl(clip_pkl, data_dir, save_pkl):
    with open(clip_pkl, 'rb') as f:
        clips_all = pickle.load(f)

    

    clip_dict = {}
    gen_dict = {}
    for mode in ['train', 'val']:
        obj_cnt = 0
        clips = clips_all[mode]
        gen_dict[mode] = []
        clip_dict[mode] = []
        for clip_ind,clip in enumerate(clips):
            clip_dir = os.path.join(data_dir, clip[1])
            txt_files = [os.path.join(clip_dir, clip[i].split('/')[-1]+'_leftImg8bit.txt') for i in range(3)]
            clip_list = []
            for txt_file in txt_files:
                with open(txt_file, 'r') as f:
                    line = f.readline().strip()
                    if line == '':
                        continue
                    objs = line.split('---')
                    obj_list = []
                    for obj in objs:
                        obj_numel = list(map(int, obj.split(',')[1:]))
                        assert len(obj_numel) == 4
                        obj_list.append(obj_numel)
                clip_list.append(obj_list)
            if len(clip_list) == 0:
                continue
            valid_ind_list = []
            for obj_id, obj in enumerate(clip_list[1]):
                if area_check(obj, thresh=0):
                    valid_ind_list.append(obj_id)
            for i in range(3):
                clip_list[i] = [clip_list[i][j] for j in valid_ind_list]
                for obj in clip_list[i]:
                    obj.insert(0, area_ratio(obj, 512, 1024))
            obj_cnt += len(valid_ind_list)
            gen_dict[mode].append(clip_list)
            clip_dict[mode].append(clip)
            sys.stdout.write('\r {}/{} {}'.format(clip_ind+1, len(clips), mode))
        print()

        print(mode)
        print(len(gen_dict[mode]))
        print('avg track obj cnt = {:.2f}'.format(float(obj_cnt)/len(clips)))

# create_trackrcnn_track_pkl('/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_extra_lsclip.pkl',
                           # '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_512x1024_trackrcnn_i9_track',
                           #  '/data/linz/proj/Dataset/Cityscape/load_files/int_9_len_3_extra_512x1024_trackrcnn_area_0_lsclip.pkl')






