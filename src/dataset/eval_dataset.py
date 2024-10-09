import os
import os.path as osp
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import  Dataset
import codecs as cs
from torch.utils.data._utils.collate import default_collate
from src.utils.word_vectorizer import WordVectorizer
import json
import pickle

class Evaluation_Dataset(Dataset):
    def __init__(self,args,split='test',mode='gt'):
        self.args=args
        self.max_motion_length=self.args.max_motion_length
        self.split=split
        self.mode=mode
        self.obj=self.args.obj # 2o or 3o
        self.data_path=self.args.data_dir

        self.obj_bps_dir=osp.join(self.data_path, "object_bps_npy_files_for_eval_joints24")
        self.obj_sampled_verts_dir=osp.join(self.data_path, "object_sampled_1024_verts")
        
        # 这个之后肯定要用上，但是现在先不用，先不考虑词表的问题
        self.w_vectorizer=WordVectorizer(osp.join(self.args.data_dir,'glove/glove_mdm'),'our_vab')
        # self.w_vectorizer=WordVectorizer(osp.join(self.args.data_dir,'glove'),'himo_vab')

        self.data=[]
        self.load_data()

        # bps和sampled_verts的信息也需要弄一弄，这里先搁置
        # self.object_bps=dict(np.load(self.obj_bps_path,allow_pickle=True)) # 1,1024,3
        # self.object_sampled_verts=dict(np.load(self.obj_sampled_verts_path,allow_pickle=True)) # 1024,3


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data=self.data[idx]
        seq_name=data['seq_name']

        # 这个data['length']在train中根本就没有这样的说法，这个是在test中才有的
        m_length=data['motion'].shape[0]
        motion=np.concatenate([data['motion'][:, :24*3], data['motion'][:, -22*6:], \
                              data['human_motion']['h_trans'], \
                                data['obj_rel_rot_6d'], data['obj_trans']], axis=-1)
        caption=data['text_annotation'][seq_name]

        # 这个实在是没有啊
        # caption,tokens=text_list['caption'],text_list['tokens']

        # if len(tokens) < self.args.max_text_len:
        #     # pad with "unk"
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens = tokens + ['unk/OTHER'] * (self.args.max_text_len + 2 - sent_len)
        # else:
        #     # crop
        #     tokens = tokens[:self.args.max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        # pos_one_hots = []
        # word_embeddings = []
        # for token in tokens:
        #     word_emb, pos_oh = self.w_vectorizer[token]
        #     pos_one_hots.append(pos_oh[None, :])
        #     word_embeddings.append(word_emb[None, :])
        # pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        # word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length<self.max_motion_length:
                # padding
                motion = np.concatenate([motion,
                                        np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                        ], axis=0)
                
        if self.mode=='gt':
            # return word_embeddings,pos_one_hots,caption,sent_len,motion,m_length,'_'.join(tokens)
            return caption,motion,m_length
        elif self.mode=='eval':
            if self.args.obj=='2o':
                obj_bps=data['obj_bps'][0:1].squeeze(0) # 1024,3
                init_state=motion[0] # 24*3+22*6+3 + 9=216

                return caption, motion, m_length,\
                    obj_bps, init_state.astype(np.float32),data["seq_name"].split('_')[1],data['betas']
            elif self.args.obj=='3o':
                obj1_bps=self.object_bps[data['obj1_name']].squeeze(0)
                obj2_bps=self.object_bps[data['obj2_name']].squeeze(0)
                obj3_bps=self.object_bps[data['obj3_name']].squeeze(0)
                init_state=motion[0]
                return word_embeddings,pos_one_hots,caption,sent_len,motion,m_length,'_'.join(tokens),\
                    obj1_bps,obj2_bps,obj3_bps,init_state.astype(np.float32),data['obj1_name'],data['obj2_name'],data['obj3_name'],data['betas']


    def load_data(self):
        with open(osp.join(self.data_path, 'test_diffusion_manip_window_300_processed_joints24.pkl'), "rb") as f:
            all_motion_dict=pickle.load(f)
        print("all_motion_dict:", len(all_motion_dict))

        # human motion
        i=0
        for index in range(len(all_motion_dict)):
            try:
                seq_motion_dict=all_motion_dict[index]
                seq_name=seq_motion_dict['seq_name']
                text_annotation_path=osp.join(self.data_path, f"text_annotations/{seq_name}.json")
                text_annotation=json.load(open(text_annotation_path))

                obj_bps_path=osp.join(self.obj_bps_dir, f"{seq_name}_{str(index)}.npy")
                obj_bps=np.load(obj_bps_path)

                obj_sampled_1024_verts_path=osp.join(self.obj_sampled_verts_dir, f"{seq_name}_1024verts_{str(index)}.npy") 
                obj_sampled_1024_verts=np.load(obj_sampled_1024_verts_path)

                obj_name=seq_name.split('_')[1]
                if obj_name in ["mop", "vacuum"]:
                    continue

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
        # print(i)
        # print(len(all_motion_dict))
        # print(len(self.data))
               
if __name__=="__main__":
    pass