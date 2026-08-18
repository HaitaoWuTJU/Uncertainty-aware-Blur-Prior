import os,mne,pickle,torch
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.discriminant_analysis import _cov
import scipy
import argparse
# Raw THINGS-EEG2 dumps: <raw_dir>/sub-XX/ses-XX/raw_eeg_{training,test}.npy
RAW_EEG_DIR = os.environ.get('UBP_RAW_EEG_DIR', '/work3/s193209/data/eeg_raw')
# Resized 224x224 images, shared across dataset formats (read-only here)
IMAGE_RESIZE_DIR = os.environ.get('UBP_IMAGE_RESIZE_DIR',
    '/work3/s193209/data/images/Image_set_Resize')
# UBP data root: EEG only (Preprocessed_data_*/ and, at train time, Image_feature/)
UBP_DATA_ROOT = os.environ.get('UBP_DATA_ROOT',
    '/work3/s193209/data/eeg_preprocessed_250hz/ubp_format')
def get_args_parser():
    parser = argparse.ArgumentParser('preprocess THINGS-EEG2 into UBP format', add_help=False)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--raw_dir', type=str, default=RAW_EEG_DIR,
        help='dir holding sub-XX/ses-XX/raw_eeg_{training,test}.npy')
    parser.add_argument('--img_dir', type=str, default=IMAGE_RESIZE_DIR,
        help='resized image dir holding train_images/ and test_images/')
    parser.add_argument('--ubp_root', type=str, default=UBP_DATA_ROOT,
        help='output root for the per-subject .pt files')
    parser.add_argument('--no_whiten', action='store_true',
        help='skip multivariate noise normalisation')
    return parser.parse_args()
args = get_args_parser()
sub = args.subject
raw_dir = args.raw_dir
ubp_root = args.ubp_root
n_ses = 4
seed = 20200220
re_sfreq= 250
tmin = -0.4
tmax = 0.8
whiten = not args.no_whiten
# Image folder must already be resized to 224x224 by preprocess/process_resize.py.
# Paths stored in the .pt files are relative to this dir, so training must be
# pointed at the same one via data.image_dir in the config.
img_root = args.img_dir
save_dir = os.path.join(ubp_root,
    f"Preprocessed_data_{re_sfreq}Hz_{'whiten' if whiten else 'no_whiten'}_minus400til800_trunc200",
    'sub-'+format(sub,'02'))
os.makedirs(save_dir, exist_ok=True)
print(f'raw_dir : {raw_dir}')
print(f'img_root: {img_root}')
print(f'save_dir: {save_dir}')
chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                  'O1', 'Oz', 'O2']
mvnn_dim = 'epochs'
def mvnn(epoched_test, epoched_train):
    
    ### Loop across data collection sessions ###
    whitened_test = []
    whitened_train = []
    for s in range(n_ses):
        session_data = [epoched_test[s], epoched_train[s]]
        ### Compute the covariance matrices ###
        # Data partitions covariance matrix of shape:
        # Data partitions × EEG channels × EEG channels
        sigma_part = np.empty((len(session_data),session_data[0].shape[2],
            session_data[0].shape[2]))
        for p in range(sigma_part.shape[0]):
            # Image conditions covariance matrix of shape:
            # Image conditions × EEG channels × EEG channels
            sigma_cond = np.empty((session_data[p].shape[0],
                session_data[0].shape[2],session_data[0].shape[2]))
            for i in tqdm(range(session_data[p].shape[0])):
                cond_data = session_data[p][i]
                # Compute covariace matrices at each time point, and then
                # average across time points
                if mvnn_dim == "time":
                    sigma_cond[i] = np.mean([_cov(cond_data[:,:,t],
                        shrinkage='auto') for t in range(cond_data.shape[2])],
                        axis=0)
                # Compute covariace matrices at each epoch (EEG repetition),
                # and then average across epochs/repetitions
                elif mvnn_dim == "epochs":
                    sigma_cond[i] = np.mean([_cov(np.transpose(cond_data[e]),
                        shrinkage='auto') for e in range(cond_data.shape[0])],
                        axis=0)
            # Average the covariance matrices across image conditions
            sigma_part[p] = sigma_cond.mean(axis=0)
        # # Average the covariance matrices across image partitions
        # sigma_tot = sigma_part.mean(axis=0)
        # ? It seems not fair to use test data for mvnn, so we change to just use training data
        sigma_tot = sigma_part[1]
        # Compute the inverse of the covariance matrix
        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)
        ### Whiten the data ###
        whitened_test.append(np.reshape((np.reshape(session_data[0], (-1,
            session_data[0].shape[2],session_data[0].shape[3])).swapaxes(1, 2)
            @ sigma_inv).swapaxes(1, 2), session_data[0].shape))
        whitened_train.append(np.reshape((np.reshape(session_data[1], (-1,
            session_data[1].shape[2],session_data[1].shape[3])).swapaxes(1, 2)
                @ sigma_inv).swapaxes(1, 2), session_data[1].shape))
    ### Output ###
    return whitened_test, whitened_train

def epoch_data(mode, sub):
    epoched_data = []
    img_conditions = []
    for s in range(n_ses):
        ### Load the EEG data and convert it to MNE raw format ###
        ses_dir = os.path.join(raw_dir, 'sub-'+format(sub,'02'),
                               'ses-'+format(s+1,'02'))
        
        # THINGS-EEG2 ships the train split as 'raw_eeg_training.npy'
        candidates = [f"raw_eeg_{'training' if mode == 'train' else mode}.npy",
                      f"raw_eeg_{mode}.npy"]
        eeg_path = next((os.path.join(ses_dir, f) for f in candidates
                         if os.path.exists(os.path.join(ses_dir, f))), None)
        if eeg_path is None:
            raise FileNotFoundError(f"none of {candidates} found in {ses_dir}")
        
        print(f'loading {eeg_path}')
        eeg_data = np.load(eeg_path, allow_pickle=True).item()
        
        ch_names = eeg_data['ch_names']
        sfreq = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data_raw = eeg_data['raw_eeg_data']
        
        # Convert to MNE raw format
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(eeg_data_raw, info)
        
        ### Get events, drop unused channels and reject target trials ###
        events = mne.find_events(raw, stim_channel='stim')
        
        # Chose all channels
        raw.pick_channels(chan_order, ordered=True)
        
        # Reject the target trials (event 99999)
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)
        
        ### Epoching, baseline correction and resampling ###
        # Extract -0.4 to 0.8s, and baseline correct using [-400ms, -200ms]
        epochs = mne.Epochs(
            raw, 
            events, 
            tmin=tmin,               # -0.4
            tmax=tmax,               # 0.8
            baseline=(tmin, -0.2),   # Baseline corrected using [-0.4, -0.2]
            preload=True
        )
        
        # Resampling
        if re_sfreq < 1000:
            epochs.resample(re_sfreq)
            
        # Truncate the -400ms to -200ms interval away, leaving exactly -200ms to 800ms.
        # include_tmax=False ensures we get exactly 250 timepoints at 250Hz.
        epochs.crop(tmin=-0.2, tmax=tmax, include_tmax=False)
        
        ch_names = epochs.info['ch_names']
        times = epochs.times

        ### Sort the data ###
        data = epochs.get_data()
        events_cond = epochs.events[:, 2]
        img_cond = np.unique(events_cond)
        
        # Select only a maximum number of EEG repetitions
        if mode == 'test':
            max_rep = 20
        else:
            max_rep = 2
            
        # Sorted data matrix of shape:
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond), max_rep, data.shape[1], data.shape[2]))
        
        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events_cond == img_cond[i])[0]
            # Randomly select only the max number of EEG repetitions
            idx = shuffle(idx, random_state=seed, n_samples=max_rep)
            sorted_data[i] = data[idx]
            
        print(f"Session {s+1} shape: {sorted_data.shape}")
        epoched_data.append(sorted_data)
        img_conditions.append(img_cond) 
        
    return epoched_data, img_conditions, ch_names, times

eeg_test,_,ch_names,times = epoch_data('test',sub)
eeg_train,img_conditions_train,_,_ = epoch_data('train',sub)
if whiten:
    whitened_test, whitened_train =  mvnn(eeg_test, eeg_train)
    del eeg_test,eeg_train
else:
    whitened_test = eeg_test
    whitened_train = eeg_train
# Derived rather than hardcoded (was (200, 80)): conditions x total repetitions
n_test_cond = whitened_test[0].shape[0]
session_list=np.zeros((n_test_cond, sum(w.shape[1] for w in whitened_test)))
for s in range(n_ses):
    if s == 0:
        merged_test = whitened_test[s]
    else:
        merged_test = np.append(merged_test, whitened_test[s], 1)
    start_index = merged_test.shape[1]-whitened_test[s].shape[1]
    end_index = merged_test.shape[1]
    session_list[:,start_index:end_index]=s
del whitened_test
# 'img': duplicated_images,
# 'label': label,
img_directory = os.path.join(img_root, 'test_images')
all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
all_folders.sort()
images = []
labels = []
texts = []
for i,folder in enumerate(all_folders):
    folder_path = os.path.join(img_directory, folder)
    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_images.sort()
    images.extend(os.path.relpath(os.path.join(folder_path, img), img_root) for img in all_images)
    labels.extend([i for img in all_images])
    texts.extend([img.rsplit('_',1)[0] for img in all_images])
n_test_reps = session_list.shape[1]
img_list = np.tile(np.array(images)[:, np.newaxis], (1, n_test_reps))
labels_list = np.tile(np.array(labels)[:, np.newaxis], (1, n_test_reps))
text_list = np.tile(np.array(texts)[:, np.newaxis], (1, n_test_reps))
print(merged_test.shape,merged_test.dtype)
print(img_list.shape)
print(labels_list.shape,labels_list.dtype)
print(img_list[0,0].split('/')[-1].rsplit('_',1)[0])
print(text_list.shape)
test_dict = {
    'eeg': merged_test.astype(np.float16),
    'label':labels_list,
    'img':img_list,
    'text':text_list,
    'session': session_list,
    'ch_names': ch_names,
    'times': times,
}
torch.save(test_dict, os.path.join(save_dir,'test.pt'),pickle_protocol=5)

### Merge and save the training data ###
# Sizes are derived from the data; they used to be hardcoded to the THINGS-EEG2
# layout ((33080, 2), white_data.shape[1]*2, (16540, 4)), which fails silently or
# loudly if a subject has a different number of sessions/repetitions.
reps_per_ses = whitened_train[0].shape[1]
ses_list = np.concatenate([np.full((w.shape[0], reps_per_ses), s)
    for s, w in enumerate(whitened_train)], axis=0)
for s in range(n_ses):
    if s == 0:
        white_data = whitened_train[s]
        img_cond = img_conditions_train[s]
    else:
        white_data = np.append(white_data, whitened_train[s], 0)
        img_cond = np.append(img_cond, img_conditions_train[s], 0)
del whitened_train
print('ses_list',len(ses_list))
unique_cond = np.unique(img_cond)
# The loop below indexes conditions as i+1, so ids must be a dense 1..N range,
# otherwise np.where returns nothing and the previous image's data is reused.
if not np.array_equal(unique_cond, np.arange(1, len(unique_cond) + 1)):
    raise ValueError('expected training condition ids 1..N, got '
        f'{unique_cond.min()}..{unique_cond.max()} ({len(unique_cond)} unique)')
cond_counts = np.bincount(img_cond)[1:]
if len(np.unique(cond_counts)) != 1:
    raise ValueError('training images appear in a varying number of sessions '
        f'({np.unique(cond_counts).tolist()}); the merge assumes a constant count')
ses_per_img = int(cond_counts[0])
total_reps = reps_per_ses * ses_per_img
print(f'{len(unique_cond)} conditions x {ses_per_img} sessions x {reps_per_ses} '
    f'reps = {total_reps} repetitions per image')
# Data matrix of shape:
# Image conditions × EGG repetitions × EEG channels × EEG time points
merged_train = np.zeros((len(unique_cond), total_reps,
    white_data.shape[2],white_data.shape[3]))
sorted_session_list = np.zeros((len(unique_cond), total_reps))
for i in range(len(unique_cond)):
    # Find the indices of the selected category
    idx = np.where(img_cond == i+1)[0]
    for r in range(len(idx)):
        sorted_session_list[i][r*reps_per_ses:(r+1)*reps_per_ses]=ses_list[idx[r]]
        if r == 0:
            ordered_data = white_data[idx[r]]
        else:
            ordered_data = np.append(ordered_data, white_data[idx[r]], 0)
    merged_train[i] = ordered_data
    
del ordered_data
img_directory = os.path.join(img_root, 'train_images')
all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
all_folders.sort()
images = []
labels = []
texts = []
for i,folder in enumerate(all_folders):
    folder_path = os.path.join(img_directory, folder)
    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_images.sort()
    images.extend(os.path.relpath(os.path.join(folder_path, img), img_root) for img in all_images)
    labels.extend([i for img in all_images])
    texts.extend([img.rsplit('_',1)[0] for img in all_images])
    
labels_list = np.tile(np.array(labels)[:, np.newaxis], (1, total_reps))
img_list = np.tile(np.array(images)[:, np.newaxis], (1, total_reps))
text_list = np.tile(np.array(texts)[:, np.newaxis], (1, total_reps))
print(merged_train.shape,merged_train.dtype)
print(labels_list.shape,labels_list.dtype)
print(img_list.shape)
print(text_list.shape)
print(sorted_session_list.shape)

train_dict = {
    'eeg': merged_train.astype(np.float16),
    'label':labels_list,
    'img':img_list,
    'text':text_list,
    'session':sorted_session_list,
    'ch_names': ch_names,
    'times': times,
}
# Create the directory if not existing and save the data
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
file_name_train = 'train.pt'
torch.save(train_dict, os.path.join(save_dir,file_name_train),pickle_protocol=5)