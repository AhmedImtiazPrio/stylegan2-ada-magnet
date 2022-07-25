import numpy as np
import warnings

class polSampler(object):
    """
    Sampler class for polarity sampling
    
    Usage:
    =====
    # For conditional sampling
    
    ## Initialize sampler
    sampler = polSampler('./svds/cifar10_pixel.npz',rho=1,top_k=10)
    
    ## Sample with initialized polarity, equal number of samples from each class
    sampled_lat,sampled_cond = sampler.sample(n=20)
    
    ## Update sampling polarity and sample conditioned on class
    sampler.update_rho(-.3)
    sampled_lat,sampled_cond = sampler.sample(n=20,conditional_class=2)
    
    # For unconditional sampling
    ## Initialize sampler
    sampler = polSampler('./svds/ffhq_pixel.npz',rho=1,top_k=10)
    
    ## Sample with initialized polarity
    sampled_lat = sampler.sample(n=20)
    
    ## Update sampling polarity
    sampler.update_rho(-.3)
    sampled_lat = sampler.sample(n=20)    
    
    """
    def __init__(self,singulars_npz_path,rho,top_k=30,is_conditional=False,verbose=True):
    """
    singulars_npz_path (str): path to numpy file containing the following:
        latents (np.ndarray): latent vectors or random seeds/states for corresponding latents
        svds (np.ndarray): singular values of the input-output transformation corresponding to latent vector locations
        labels (np.ndarray): optional, one hot encoded labels for conditional generators
        
    rho (float): polarity parameter rho
    top_k (int): top singulars to use for volume scalar estimation
    is_conditional (bool): conditional generator flag
    """    
        
        with np.load(singulars_npz_path) as data:
            
            self.svds = data['svds']
            self.latents = data['latents']
            
            self.labels = None
            
            if 'labels' in [each for each in data.keys()]: ## if has stored condition vectors, legacy support
                
                self.labels = data['labels']
                
                if self.labels == []:  ## hasattr but attr is []
                    self.labels = None
                    assert not is_conditional ## condition data not found in npz
                else:
                    is_conditional = True ## condition data found in npz
                    warnings.warn('Condition data found in npz file, setting `is_conditional=True`')
                    
        self.is_conditional = is_conditional
        self.top_k = top_k
        self.rho = rho
        
        # for conditional
        if labels is not None:
            
            if not labels.shape[1]>1:
                raise('Labels should be one hot encoded')
                
            self.class_labels = np.argmax(labels,axis=-1)
            self.unique_classes = np.unique(self.class_labels)
            self.num_classes = labels.shape[1]
            
        else:
            self.classes = None
        
        ## get unnormalized probabilities
        sigma = np.exp(np.log(self.svds[:,:self.top_k].astype(np.float64)).sum(1))
        proba_un = sigma**self.rho
        proba_un = np.clip(proba_un,1e-200,1e200)
        
        ## if conditional normalize per class and maintain separate class proba
        if not self.is_conditional:
            self.proba = proba_un/proba_un.sum()
            
        else:
            self.proba = []
            self.classwise_idx = []
            for each_class in self.unique_classes:
                idxs = np.where(self.class_labels == each_class)[0]
                self.classwise_idx.append(idxs)
                self.proba.append(proba_un[idxs]/np.sum(proba_un[idxs]))
        
    def sample(self,n,replace=False,conditional_class=None,seed=None):
        """
        Sample latents (and condition vectors) using precomputed weights
        
        n (int): number of samples
        replace (bool): sample with/without replacing
        conditional_class (int): optional, class to use for sampling
        seed (int): optional, random seed for sampling
        """
        
        np.random.seed(seed)
        
        if self.is_conditional:
            
            if conditional_class is None: # condition not specified
                
                assert n % self.num_classes == 0 ## to sample equal number from each class
                
                latents = []
                labels = []
                for i in range(self.num_classes):
                    
                    sample_index = np.random.choice(
                                    self.classwise_idx[i],
                                    size=n//self.num_classes,
                                    p= self.proba[i],
                                    replace=replace
                                    )
                    
                    latents.append(self.latents[sample_index])
                    labels.append(self.labels[sample_index])
                
                return np.vstack(latents),np.vstack(labels)
            
            else: # conditional class is specified
                
                sample_index = np.random.choice(
                                    self.classwise_idx[conditional_class],
                                    size=n,
                                    p= self.proba[conditional_class],
                                    replace=replace
                                    )
                
                return self.latents[sample_index],self.labels[sample_index]
                
        else: ## non-conditional
            
            sample_index = np.random.choice(
                                    self.latents.shape[0],
                                    size=n,
                                    p= self.proba,
                                    replace=replace
                                    )
                
            return self.latents[sample_index]

        
    def update_rho(self,rho):
        """
        Recompute probabilites for new rho/polarity
        """
        self.rho = rho
        
        sigma = np.exp(np.log(self.svds[:,:self.top_k].astype(np.float64)).sum(1))
        proba_un = sigma**self.rho
        proba_un = np.clip(proba_un,1e-200,1e200)
        
        ## if conditional normalize per class and maintain separate class proba
        if not self.is_conditional:
            self.proba = proba_un/proba_un.sum()
            
        else:
            self.proba = []
            self.classwise_idx = []
            for each_class in self.unique_classes:
                idxs = np.where(self.class_labels == each_class)[0]
                self.classwise_idx.append(idxs)
                self.proba.append(proba_un[idxs]/np.sum(proba_un[idxs]))
                
def to_uint8(img):
    '''
    Convert image to uint8
    '''
    img = img*127.5 + 128
    img = np.clip(img,0,255)
    return img.astype(np.uint8) 

def imgrid(imarray, cols=10, pad=1, pad_value=0):
    '''
    Display image array in a grid
    
    imarray (np.uint8): array of numpy uint8 images
    cols (int): number of columns in final image grid
    pad (int): pixel padding
    pad_value (int): 0-255, value to pad with
    '''
    if imarray.dtype != np.uint8:
        raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = N // cols + int(N % cols != 0)
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=pad_value)
    H += pad
    W += pad
    grid = (imarray
        .reshape(rows, cols, H, W, C)
        .transpose(0, 2, 1, 3, 4)
        .reshape(rows*H, cols*W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid