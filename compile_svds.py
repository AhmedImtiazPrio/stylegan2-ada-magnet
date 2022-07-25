import tqdm
import numpy as np
import sys

if len(sys.argv)<3:
    raise AttributeError('''
    Usage:
    `python save_filename conditional_flag regexp_path_to_npz`
    
    save_filename: filename to save compiled data to
    conditional_flag: binary flag showing whether conditional data or not
    regexp_path_to_npz: regular expression for npz files to compile together
    ''')
    
save_file = sys.argv[1]
conditional_flag = sys.argv[2]
filenames = sys.argv[3:]

svds = []
latents = []
labels = []

for each_file in tqdm.tqdm(filenames):
    with np.load(each_file) as data:
        svd = data['svds']
        latent = data['latents']
        
        if conditional_flag:
            label = data['labels']
        
    svds.append(svd.reshape(-1,svd.shape[-1]))
    latents.append(latent.reshape(-1,latent.shape[-1]))
    
    if conditional_flag:
        labels.append(label.reshape(-1,label.shape[-1]))
    
    
svds = np.concatenate(svds,axis=0)
latents = np.concatenate(latents,axis=0)

if conditional_flag:
    labels = np.concatenate(labels,axis=0)

print(svds.shape)
print(latents.shape)
print(labels.shape)

np.savez(save_file,latents=latents,svds=svds,labels=labels)
    