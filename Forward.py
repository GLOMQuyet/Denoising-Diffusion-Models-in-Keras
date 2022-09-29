import tensorflow as tf


class Forward_Noise(tf.keras.layers.Layer):

  def __init__(self,sqrt_alpha_bar,one_minus_sqrt_alpha_bar,**kwargs):
    super(Forward_Noise,self).__init__(**kwargs)
    self.sqrt_alpha_bar = sqrt_alpha_bar
    self.one_minus_sqrt_alpha_bar = one_minus_sqrt_alpha_bar

  def forward_noise(self,x_0,t):
    noise = tf.random.normal(x_0.shape)
    reshaped_sqrt_alpha_bar_t = tf.cast(tf.experimental.numpy.reshape(tf.experimental.numpy.take(self.sqrt_alpha_bar, t), (-1, 1, 1, 1)),tf.float32)
    reshaped_one_minus_sqrt_alpha_bar_t = tf.cast(tf.experimental.numpy.reshape(tf.experimental.numpy.take(self.one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1)),tf.float32)
    noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
    return noisy_image,noise

  def call(self,x_0,t):
    noise_img,noise = self.forward_noise(x_0,t)
    return noise_img,noise


class Generator(tf.keras.layers.Layer):
  def __init__(self, timesteps, **kwargs):
    super(Generator, self).__init__()
    self.timesteps = timesteps

  def generate_timestamp(self, num):
    return tf.random.uniform(shape=[num], minval=0, maxval=self.timesteps, dtype=tf.int32)

  def call(self, x):
    x = self.generate_timestamp(x)
    return x




