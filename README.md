## Plug-and-play codes for StyleGAN2-ada with Polarity Sampling
### Polarity Sampling: Quality and Diversity Control of Pre-Trained Generative Networks via Singular Values, CVPR 2022 (Oral)
### [Paper Link](https://arxiv.org/abs/2203.01993) | [Video Link](https://www.youtube.com/watch?v=zRKyx_dF89M)

*Authors:* Ahmed Imtiaz Humayun, Randall Balestriero, Richard Baraniuk

*Abstract:* We present Polarity Sampling, a theoretically justified plug-and-play method for controlling the generation quality and diversity of pre-trained deep generative networks DGNs). Leveraging the fact that DGNs are, or can be approximated by, continuous piecewise affine splines, we derive the analytical DGN output space distribution as a function of the product of the DGN's Jacobian singular values raised to a power ρ. We dub ρ the polarity parameter and prove that ρ focuses the DGN sampling on the modes (ρ<0) or anti-modes (ρ>0) of the DGN output-space distribution. We demonstrate that nonzero polarity values achieve a better precision-recall (quality-diversity) Pareto frontier than standard methods, such as truncation, for a number of state-of-the-art DGNs. We also present quantitative and qualitative results on the improvement of overall generation quality (e.g., in terms of the Frechet Inception Distance) for a number of state-of-the-art DGNs, including StyleGAN3, BigGAN-deep, NVAE, for different conditional and unconditional image generation tasks. In particular, Polarity Sampling redefines the state-of-the-art for StyleGAN2 on the FFHQ Dataset to FID 2.57, StyleGAN2 on the LSUN Car Dataset to FID 2.27 and StyleGAN3 on the AFHQv2 Dataset to FID 3.95. Demo: [bit.ly/polarity-samp](http://bit.ly/polarity-samp)

### Repository with plug-and-play codes for a number of different models/datasets at [magnet-polarity](https://github.com/AhmedImtiazPrio/magnet-polarity) 

## Setup

The repository is based on the official Tensorflow StyleGAN2-ada repository. The additional plug-and-play elements in the repository are the following:
1. `get_svds.py`: To calculate the singular values required to estimate volume scalars
2. `compile_svds.py`: `get_svds.py` creates multiple numpy files with latent vectors and corresponding singular values. `compile_svds.py` compiles them together for downstream tasks.
3. `polarity_utils.py`: Contains the `polSampler` class that uses precomputed singular values for polarity sampling.

### To get started:

```
git clone https://github.com/AhmedImtiazPrio/stylegan2-ada-magnet.git
```


### Requirements:
```
tensorflow-gpu==1.15
numpy
```

## Usage
1. Download network weight .pkl file from [here](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/). For example, for cifar10:
```shell
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl
```

2. Calculate singular values for random latents

```shell
python get_svds.py --network=cifar10.pkl --N=200000 --label_size=10 --proj_dim=128 --save_path=./svds/
```

3. Compile calculated svds. Specify `save_path`, `conditional_flag` and list .npz files to compile 
```shell
python compile_svds.py cifar10_singulars.npz 1 ./svds/*.npz
```

4. Sample latents
```python
from polarity_utils import polSampler

sampler = polSampler('cifar10_singulars.npz',rho=1,top_k=15,is_conditional=True)
latents, labels = sampler.sample(n=100)

sampler.update_rho(-.3)
latents, labels = sampler.sample(n=100)
```

5. Generate images using latents
```python
from dnnlib import tflib
import tensorflow as tf

import matplotlib.pyplot as plt
from polarity_utils import imgrid, to_uint8

with tf.Graph().as_default(), tflib.create_session().as_default():
    
    with dnnlib.util.open_url('cifar10.pkl') as f:
        _G, _D, Gs = pickle.load(f)
    
    imgs = []
    for i in range(latents.shape[0]):
        
        latents_in = tf.convert_to_tensor(latents[i][None,...])
        labels_in = tf.convert_to_tensor(labels[i][None,...])

        images = Gs.get_output_for(latents_in, labels_in,
                                truncation_psi=1, is_validation=True)

        imgs.append(to_uint8(dnnlib.tflib.run(images).transpose(0,2,3,1)))
        
plt.figure()
plt.imshow(imgrid(np.vstack(imgs),cols=10))
```


### Citation
```
@InProceedings{Humayun_2022_polarity,
    author    = {Humayun, Ahmed Imtiaz and Balestriero, Randall and Baraniuk, Richard},
    title     = {Polarity Sampling: Quality and Diversity Control of Pre-Trained Generative Networks via Singular Values},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10641-10650}
}
```
