import os
import os.path as osp
import torch
import h5py
import sys
from tqdm import tqdm
from torch.utils.data import Dataset
from src.utils.misc import to_tensor
import numpy as np
import pickle
import json
import smplx

from scipy.spatial.transform import Rotation as R

class HIMO_2O(Dataset):
    def __init__(self,
                args,
                split='train'):
        super(HIMO_2O,self).__init__()
        self.args=args
        self.split=split

        self.data_path=self.args.data_dir # bps和sampled_verts的路径也应该放在这里
        self.object_bps_dir=osp.join(self.data_path,"object_bps_npy_files_joints24")
        self.object_sampled_verts_dir=osp.join(self.data_path,"object_sampled_1024_verts")

        self.max_frames=self.args.num_frames # 定义是300

        self.data=[]
        
        self.load_data()
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq_data=self.data[idx]
        m_len=seq_data['motion'].shape[0]   # seq len
        
        human_motion=np.concatenate(
            [
                seq_data['motion'][:,:24*3],    # global joint position
                seq_data['motion'][:, -22*6:],  # global joint rotation
                seq_data['human_motion']['h_trans'] # global translation
            ], axis=-1 # nf,24*3+22*6+3 = 207
        )

        # 物体trans
        obj_trans=seq_data['obj_trans'] # nf,3
        obj_rel_rot_6d=seq_data['obj_rel_rot_6d'] # nf,6

        x_start=np.concatenate(
            [
                human_motion,
                obj_rel_rot_6d,
                obj_trans
            ], axis=-1 # nf,24*3+22*6+3 +6+3 = nf, 216
        )

        seq_name=seq_data['seq_name']
        text=seq_data["text_annotation"][seq_name].strip() # caption，因为之前直接死load json了，所以这里需要seq_name

        if m_len<self.max_frames:
            x_start=np.concatenate([x_start,np.zeros([self.max_frames-m_len,x_start.shape[1]])],axis=0)

        # object bps and sampled verts
        obj_bps=seq_data['obj_bps'][0:1] # 1,1024,3
        obj_sampled_verts=seq_data['obj_sampled_1024_verts'][0]    # 1024,3

        init_state=x_start[0] # 24*3+22*6+3+6+3 = 216

        betas=seq_data['betas'][0] # 16

        return text, obj_bps, obj_sampled_verts, init_state, x_start, m_len, betas


    def load_data(self):
        with open(osp.join(self.data_path,"train_diffusion_manip_window_300_cano_joints24.pkl"), "rb") as f:
            all_motion_dict=pickle.load(f)
        print("all_motion_dict:", len(all_motion_dict))

        # human motion
        i=0
        for index in range(len(all_motion_dict)):
            try:
                seq_motion_dict=all_motion_dict[index]
                seq_name=seq_motion_dict['seq_name']

                obj_name=seq_name.split('_')[1]
                if obj_name in ["mop", "vacuum"]:
                    continue

                text_annotation_path=osp.join(self.data_path, f"text_annotations/{seq_name}.json")
                text_annotation=json.load(open(text_annotation_path))

                obj_bps_path=osp.join(self.object_bps_dir, f"{seq_name}_{str(index)}.npy")
                obj_bps=np.load(obj_bps_path)

                obj_sampled_1024_verts_path=osp.join(self.object_sampled_verts_dir, f"{seq_name}_1024verts_{str(index)}.npy") 
                obj_sampled_1024_verts=np.load(obj_sampled_1024_verts_path)

                seq_motion_dict.update(
                    {
                        'text_annotation': text_annotation,
                        'obj_bps': obj_bps, # nf,1024,3
                        'obj_sampled_1024_verts': obj_sampled_1024_verts    # nf,1024,3
                        }
                )

                self.data.append(seq_motion_dict)
            except FileNotFoundError as e:
                i+=1
                # print(e)
                continue

        # print("self.data:", len(self.data))
        # print("i:", i)
                