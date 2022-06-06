# from training import misc

import dnnlib.tflib
from dnnlib import tflib
import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.python.ops.parallel_for import batch_jacobian
from datetime import datetime
import tqdm
import sys
import pickle
import os

minibatch_per_gpu = int(sys.argv[1])
num_gpu = int(sys.argv[2])
truncation_psi = 1
num_latents = int(sys.argv[3])
network_path = str(sys.argv[4])
save_path = str(sys.argv[5])

if not os.path.exists(save_path):
    os.mkdir(save_path)

with tf.Graph().as_default(), tflib.create_session().as_default():
    
    with dnnlib.util.open_url(network_path) as f:
        _G, _D, Gs = pickle.load(f)
    
    proj_svd = 128
    output_flatten = np.prod(Gs.output_shape[1:])
    
    init = tf.keras.initializers.Orthogonal(gain=1.0, seed=0)
    w = init(shape=(proj_svd, output_flatten))
    
    result_expr = []
    
    for gpu_idx in range(num_gpu):
            
        with tf.device('/gpu:%d' % gpu_idx):

            Gs_clone = Gs.clone()

            latents_in = tf.random_normal([minibatch_per_gpu] + Gs_clone.input_shape[1:])
            labels = None
            
            images = Gs_clone.get_output_for(latents_in, labels,
                                             truncation_psi=truncation_psi, is_validation=True)

            images = tf.reshape(images,shape=(minibatch_per_gpu,output_flatten))
            proj_image = tf.matmul(images, w, transpose_b=True)

            jacobian = batch_jacobian(proj_image, latents_in, use_pfor=False)
            svd = tf.linalg.svd(jacobian, full_matrices=False, compute_uv=False)

            result_expr.append([svd,latents_in])

    
    for it in tqdm.tqdm(range(0,num_latents//1000),desc='saving loop'):
        
        svds = []
        latents = []

        for itt in tqdm.tqdm(range(1000),desc='sampling loop'):
            
            res = tflib.run(result_expr)
            
            for each in res:
                svds.append(np.asarray(each[0]))
                latents.append(np.asarray(each[1]))

        print('saving...')
        latents = np.stack(latents)
        print(latents.shape)
        print(f'Latents range {latents.max()} {latents.min()}')
        svds = np.stack(svds)
        print(svds.shape)
        print(f'SVDS range {svds.max()} {svds.min()}')

        np.savez(
            os.path.join(save_path,f"{''.join(network_path.split('.')[:-1])}_{it}_{str(datetime.now()).replace(' ','-')}.npz"),
            svds=svds,
            latents=latents
        )