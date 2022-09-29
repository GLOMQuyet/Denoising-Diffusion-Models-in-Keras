import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from model import UNet
from Forward import Forward_Noise,Generator
from Dataset import DataGenerator
from backward import Backward_Noise
# HYPARAMETER

# data
num_epochs = 0  # train for at least 50 epochs for good results
image_size = 32

# optimization
batch_size = 256
learning_rate = 1e-3
weight_decay = 1e-4

# model
timesteps = 500
a_min = 0.0001
a_max = 0.02
net = UNet(c_out=1)
# create a fixed beta schedule
beta = np.linspace(a_min,a_max, timesteps+1)

# this will be used as discussed in the reparameterization trick
alpha = 1 - beta
alpha_bar = np.cumprod(alpha, 0)
alpha_bar = np.concatenate((np.array([1.]), alpha_bar[:-1]), axis=0)
sqrt_alpha_bar = np.sqrt(alpha_bar)
one_minus_sqrt_alpha_bar = np.sqrt(1-alpha_bar)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

loss_fn = tf.keras.losses.MeanSquaredError()
# Prepare the metrics.
train_acc_metric = tf.keras.metrics.MeanSquaredError('mse train')
val_acc_metric = tf.keras.metrics.MeanSquaredError('mse val')
# Optimizers
opt = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)

# Diffussion model
forward_noise = Forward_Noise(sqrt_alpha_bar,one_minus_sqrt_alpha_bar)
train_generator = DataGenerator(x_train, image_size = image_size,batch_size =batch_size,shuffle=True)
val_generator = DataGenerator(x_test,image_size = image_size,batch_size=batch_size,shuffle=True)
generate_timestamp = Generator(timesteps)
ddpm = Backward_Noise(alpha,alpha_bar,beta)

# Load checkpoint model
# Checkpoint Hyparameter
ckpt = tf.train.Checkpoint(net=net)
ckpt_manager = tf.train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=2)
# load from a previous checkpoint if it exists, else initialize the model from scratch

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    start_interation = int(ckpt_manager.latest_checkpoint.split("-")[-1])
    print("Restored from {}".format(ckpt_manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


