class polSampler(object):
    """
    Sampler class for polarity sampling
    latents (np.ndarray): latent vectors or random seeds for corresponding latents
    svds (np.ndarray): singular values of the input-output transformation corresponding to latent vector locations
    labels (np.ndarray): one hot encoded labels for conditional generators
    """
    def __init__(self,svds,latents,rho,top_k=30,labels=None):
        
        self.svds = svds
        self.latents = latents
        self.top_k = top_k
        self.rho = rho
        self.labels = labels
        
        # for conditional
        if labels is not None:
            
            if not labels.shape[1]>1:
                raise('Labels should be one hot encoded')
                
            self.class_labels = np.argmax(labels,axis=-1)
            self.unique_classes = np.unique(self.class_labels)
            self.num_classes = labels.shape[1]
            self.is_conditional = True
            
        else:
            self.classes = None
            self.is_conditional = False
        
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
        Uses precomputed probabilites based on initialized rho
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
        Recomputes probabilites for new rho
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