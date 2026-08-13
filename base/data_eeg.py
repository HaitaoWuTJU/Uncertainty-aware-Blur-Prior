import torch,os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import logging
import open_clip
import gc
from tqdm import tqdm
import itertools
import copy
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from base.utils import instantiate_from_config, get_device 


def load_eeg_data(config):
    exp_setting = config.get('exp_setting', 'intra-subject')
    # Upstream hardcoded 32/25 workers, which oversubscribes a batch job that was
    # allocated fewer cores; data.num_workers overrides both when set.
    nw = config['data'].get('num_workers')
    train_nw = 32 if nw is None else nw
    test_nw = 25 if nw is None else nw

    if exp_setting == 'intra-subject':
        test_dataset = EEGDataset(config,mode='test')
        print('init test_dataset success')
        train_dataset = EEGDataset(config,mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False,num_workers=test_nw, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=train_nw, pin_memory=True)
        return train_loader, test_loader,test_loader
    
    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = EEGDataset(config,mode='test')
        print('init test_dataset success')
        
        all_subjects = [f'sub-{i:02}' for i in range(1, 11)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = EEGDataset(leave_one_subjects_config,mode='test')
        print('init val_dataset success')
        train_dataset = EEGDataset(leave_one_subjects_config,mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False,num_workers=test_nw)#, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=False, drop_last=False,num_workers=train_nw)#, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=train_nw)#, pin_memory=True)
        return train_loader, val_loader, test_loader
    
class EEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config= config
        self.data_dir = config['data']['data_dir']
        # Resized images and the CLIP feature cache need not sit next to the EEG .pt
        # files; fall back to the upstream <data_dir>/../ layout when unset.
        self.image_dir = config['data'].get('image_dir') or os.path.join(self.data_dir, '..', 'Image_set_Resize')
        self.feature_dir = config['data'].get('feature_dir') or os.path.join(self.data_dir, '..', 'Image_feature')
        # self.img_directory = os.path.join(self.data_dir,'../','Image_set_Resize',f'{mode}_images')
        # self.all_class_names = [d.split('_',1)[-1] for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        # self.all_class_names.sort()
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                        'O1', 'Oz', 'O2']
        if self.selected_ch == "None":
            self.selected_ch = self.channels
    
        self.avg = config['data'][f"{mode}_avg"]

        self.blur_type = config['data']['blur_type']

        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode=='train' else 200
        self.per_trials = 4 if self.mode=='train' else 80

        self.data_paths = [os.path.join(self.data_dir,subject,f'{mode}.pt') for subject in self.subjects]
        self.loaded_data= [self.load_data(data_path) for data_path in self.data_paths]
        
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject*len(self.subjects)

        data_dir = os.path.join(self.feature_dir,f"{config['data']['blur_type']['target'].rsplit('.',1)[-1]}")
        os.makedirs(data_dir,exist_ok=True)

        # Image features depend only on the vision backbone and the blur settings --
        # not on the brain backbone, channel subset or trial averaging. feature_name
        # lets those variants share one cache instead of re-encoding every image.
        feature_name = config['data'].get('feature_name') or self.name
        features_filename = os.path.join(data_dir,f"{feature_name}_{mode}.pt")
        print(f'image feature cache: {features_filename}')

        pretrain_map= {
                'RN50':{'pretrained':'openai','resize':(224,224)}, #1024 
                'RN101':{'pretrained':'openai','resize':(224,224)}, #512
                'ViT-B-16':{'pretrained':'laion2b_s34b_b88k','resize':(224,224)}, #512
                'ViT-B-32':{'pretrained':'laion2b_s34b_b79k','resize':(224,224)}, #512
                'ViT-L-14':{'pretrained':'laion2b_s32b_b82k','resize':(224,224)}, #768
                'ViT-H-14':{'pretrained':'laion2b_s32b_b79k','resize':(224,224)}, #1024
                'ViT-g-14':{'pretrained':'laion2b_s34b_b88k','resize':(224,224)}, #1024
                'ViT-bigG-14':{'pretrained':'laion2b_s39b_b160k','resize':(224,224)}, #1280
            }

        self.c = config['c']
        if self.config['data']['uncertainty_aware']:
            self.blur_transform = {}
            for shift,tag in zip([-self.c,0,self.c],['low','medium','high']):
                blur_param = copy.deepcopy(config['data']['blur_type'])
                blur_param['params']['blur_kernel_size'] += shift
                self.blur_transform[tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config['data']['blur_type'])
        process_term = [transforms.ToTensor(), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))] #transforms.Resize(pretrain_map[self.model_type]['resize']), 
        self.process_transform = transforms.Compose(process_term)

        self.match_label = np.ones(self.trial_all_subjects, dtype=int)

        if  os.path.exists(features_filename):
            saved_features = torch.load(features_filename, weights_only=False)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features['text_features']
        else:
            device = get_device('auto')
            self.vlmodel, self.preprocess,_ = open_clip.create_model_and_transforms(self.model_type, device=f"cuda:{device}",pretrained=pretrain_map[self.model_type]['pretrained'])
            for param in self.vlmodel.parameters():
                param.requires_grad = False
            self.vlmodel.eval()
            if self.config['data']['uncertainty_aware']:
                self.img_features = {}
                for tag in ['low','medium','high']:
                    self.img_features[tag] = self.ImageEncoder(self.loaded_data[0]['img'],self.blur_transform[tag])
                self.img_features['avg'] = {k: (sum(self.img_features[tag][k] for tag in ['low', 'medium', 'high']) / 3) for k in self.img_features['medium']}
            else:
                self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])
            self.text_features = self.Textencoder(self.loaded_data[0]['text'])
            # The cache path has no subject in it, so concurrent runs (e.g. an LSF
            # job array over subjects) all target the same file. Write to a unique
            # temp file and rename: os.replace is atomic, so no reader ever sees a
            # half-written cache. Warm the cache with a single run to avoid the
            # redundant encoding work entirely.
            tmp_filename = f'{features_filename}.tmp.{os.getpid()}'
            torch.save({
                'text_features': self.text_features,
                'img_features': self.img_features,
            }, tmp_filename)
            os.replace(tmp_filename, features_filename)

            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    def load_data(self,data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ',1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg']=torch.from_numpy(loaded_data['eeg'])
        
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:,:,selected_idx]
        if self.avg:
            avg_data={}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:,0]
            avg_data['img'] = loaded_data['img'][:,0]
            avg_data['text'] = loaded_data['text'][:,0]
                
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1,*loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data
        
        
        for k,v in loaded_data.items():
            if k in ['eeg','label','img','text','session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data    
    
    @torch.no_grad()
    def ImageEncoder(self,images,blur_transform=None):
        if blur_transform == None:
            blur_transform = self.blur_transform
        self.vlmodel.eval()

        set_images = list(set(images))
        set_images.sort()
        batch_size = 128
        image_features_list = []
        for i in tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]

            device = next(self.vlmodel.parameters()).device
            ele = [self.process_transform(blur_transform(Image.open(os.path.join(self.image_dir,img)).convert("RGB"))) for img in batch_images]

            image_inputs = torch.stack(ele).to(device)

            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features/batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]:image_features[i].float().cpu() for i in range(len(set_images))}
        return image_features_dict
    
    @torch.no_grad()
    def Textencoder(self, text):   
        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f"This is a {t}.") for t in set_text])
        device = next(self.vlmodel.parameters()).device
        text_inputs =  text_inputs.to(device)
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        text_features_dict = {set_text[i]:text_features[i].float().cpu() for i in range(len(set_text))}
        return text_features_dict
    
    def __getitem__(self, index):
        
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index//self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]

        img = 'None' #Image.open(os.path.join(self.data_dir,'../Image_set_Resize',img_path)).convert("RGB")
    
        match_label = self.match_label[index]
        
        if self.config['data']['uncertainty_aware']:
            if self.mode == 'train':
                if match_label==0:
                    tag='low'
                elif match_label==2:
                    tag='high'
                else:
                   tag='medium'
            else:
                tag='medium'
            img_features = self.img_features[tag][img_path]
        else:
            img_features = self.img_features[img_path]

        text =  f"This is a {self.loaded_data[subject]['text'][trial_index]}."
        text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
        session = self.loaded_data[subject]['session'][trial_index]
        
        sample  = {
            'idx': index,
            'eeg': eeg[:,self.timesteps[0]:self.timesteps[1]],
            'label': label,
            'img_path': img_path,
            'img': img,
            'img_features': img_features,
            'text': text,
            'text_features': text_features,
            'session': session,
            'subject': subject,
            'eeg_mean': eeg_mean[:,self.timesteps[0]:self.timesteps[1]],
        }
        return sample
    
    def __len__(self):
        return self.trial_all_subjects
    
