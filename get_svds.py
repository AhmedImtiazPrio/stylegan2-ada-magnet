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
import argparse


parser = argparse.ArgumentParser(
        description='calculate singular values for a given pretrained model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
parser.add_argument('--batch_size', help='minibatch per gpu', type=int, default=16)
parser.add_argument('--num_gpu', help='number of gpus', type=int, default=1)
parser.add_argument('--truncation_psi', help='truncation for stylegan', type=float, default=1)
parser.add_argument('--N', help='number of latents to sample', type=int, required=True)
parser.add_argument('--network', help='path to pretrained network pkl', type=str, required=True)
parser.add_argument('--save_path', help='path to save svds', type=str, required=True)
parser.add_argument('--label_size', help='number of labels for conditional sampling', type=int, default=0)
parser.add_argument('--proj_dim', help='projection dimensions', type=int, default=128)

args = parser.parse_args()

minibatch_per_gpu = args.batch_size
num_gpu = args.num_gpu
truncation_psi = args.truncation_psi
num_latents = args.N
network_path = args.network
save_path = args.save_path
label_size = args.label_size
proj_svd = args.proj_dim

# minibatch_per_gpu = int(sys.argv[1])
# num_gpu = int(sys.argv[2])
# truncation_psi = 1
# num_latents = int(sys.argv[3])
# network_path = str(sys.argv[4])
# save_path = str(sys.argv[5])

if not os.path.exists(save_path):
    os.mkdir(save_path)

with tf.Graph().as_default(), tflib.create_session().as_default():
    
    with dnnlib.util.open_url(network_path) as f:
        _G, _D, Gs = pickle.load(f)
    
#     proj_svd = 128
    output_flatten = np.prod(Gs.output_shape[1:])
    
    init = tf.keras.initializers.Orthogonal(gain=1.0, seed=0)
    w = init(shape=(proj_svd, output_flatten))
    
    result_expr = []
    
    for gpu_idx in range(num_gpu):
            
        with tf.device('/gpu:%d' % gpu_idx):

            Gs_clone = Gs.clone()

            latents_in = tf.random_normal([minibatch_per_gpu] + Gs_clone.input_shape[1:])
            labels_in = tf.zeros([minibatch_per_gpu, 0])
            
            if label_size:
                labels_in = tf.gather(tf.eye(label_size), tf.random.uniform([minibatch_per_gpu], 0, label_size, dtype=tf.int32))
                
            images = Gs_clone.get_output_for(latents_in, labels_in,
                                             truncation_psi=truncation_psi, is_validation=True)

#             result_expr.append([images,latents_in,labels_in]) ## debug
            
            images = tf.reshape(images,shape=(minibatch_per_gpu,output_flatten))
            proj_image = tf.matmul(images, w, transpose_b=True)

            jacobian = batch_jacobian(proj_image, latents_in, use_pfor=False)
            svd = tf.linalg.svd(jacobian, full_matrices=False, compute_uv=False)

            result_expr.append([svd,latents_in,labels_in])

    
    for it in tqdm.tqdm(range(0,num_latents//1000),desc='saving loop'):
        
        svds = []
        latents = []
        labels = []

        for itt in tqdm.tqdm(range(1000),desc='sampling loop'):
            
            res = tflib.run(result_expr)
            
            for each in res:
                svds.append(np.asarray(each[0]))
                latents.append(np.asarray(each[1]))
                
                if label_size:
                    labels.append(np.asarray(each[2]))

        print('saving...')
        latents = np.stack(latents)
        
        if label_size:
            labels = np.stack(labels)
        
        print(latents.shape)
        print(f'Latents range {latents.max()} {latents.min()}')
        svds = np.stack(svds)
        print(svds.shape)
        print(f'SVDS range {svds.max()} {svds.min()}')

        np.savez(
            os.path.join(save_path,f"{''.join(network_path.split('.')[:-1])}_{it}_{str(datetime.now()).replace(' ','-')}.npz"),
            svds=svds,
            latents=latents,
            labels=labels
        )
