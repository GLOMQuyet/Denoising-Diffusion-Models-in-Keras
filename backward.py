import tensorflow as tf
from config import *


class Backward_Noise(tf.keras.layers.Layer):
    def __init__(self, alpha, alpha_bar, beta, **kwargs):
        super(Backward_Noise, self).__init__()
        self.alpha = alpha
        self.alpha_bar = alpha_bar
        self.beta = beta

    def ddpm(self, x_t, pred_noise, t):
        alpha_t = tf.cast(tf.experimental.numpy.take(self.alpha, t), tf.float32)
        alpha_t_bar = tf.cast(tf.experimental.numpy.take(self.alpha_bar, t), tf.float32)

        eps_coef = (1 - alpha_t) / (1 - alpha_t_bar) ** .5

        mean = (1 / (alpha_t ** .5)) * (x_t - eps_coef * pred_noise)

        var = tf.cast(tf.experimental.numpy.take(self.beta, t), tf.float32)
        z = tf.random.normal(x_t.shape)

        return mean + (var ** .5) * z

    def call(self, x_t, pred_noise, t):
        x = self.ddpm(x_t, pred_noise, t)
        return x