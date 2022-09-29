import tensorflow as tf
import tensorflow_addons as tfa

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=self.channels)
        self.ln = tf.keras.layers.LayerNormalization()
        self.ff_self = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(channels),
            tfa.layers.GELU(),
            tf.keras.layers.Dense(channels)
        ]
        )

    def build(self, x):
        self.reshape1 = tf.keras.layers.Reshape((x[1] * x[2], self.channels))
        self.reshape2 = tf.keras.layers.Reshape((x[1], x[2], self.channels))

    def call(self, x):
        x = self.reshape1(x)
        x_ln = self.ln(x)

        attention_value = self.mha(x_ln, x_ln, x_ln)

        attention_value = attention_value + x

        attention_value = self.ff_self(attention_value) + attention_value

        return self.reshape2(attention_value)


class DoubleConv(tf.keras.layers.Layer):
    def __init__(self, out_channels, mid_channels=None, residual=False):
        super(DoubleConv, self).__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(mid_channels, kernel_size=3, padding='same', use_bias=False),
            tfa.layers.GroupNormalization(1),
            tfa.layers.GELU(),
            tf.keras.layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False),
            tfa.layers.GroupNormalization(1)
        ]

        )
        self.gelu = tfa.layers.GELU()

    def call(self, x):
        if self.residual:
            return self.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class SILU(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(SILU,self).__init__()
        self.sigmoid = tf.keras.layers.Activation("sigmoid")
    def call(self,x):
        theta = 1.0
        return x* self.sigmoid(theta*x)


class Down(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.maxpool_conv = tf.keras.Sequential(
            [
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                DoubleConv(in_channels, residual=True),
                DoubleConv(out_channels)
            ]
        )

        self.emb_layer = tf.keras.Sequential([
            SILU(),
            tf.keras.layers.Dense(out_channels)]
        )

    def repeat(self, t, x):
        t = t[:, None, None, :]

        t = tf.repeat(t, [x.shape[1]], axis=1)
        t = tf.repeat(t, [x.shape[2]], axis=2)
        return t

    def call(self, x, t):
        x = self.maxpool_conv(x)
        t = self.emb_layer(t)
        emb = self.repeat(t, x)

        return x + emb

class Up(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(Up,self).__init__()

        self.up = tf.keras.layers.UpSampling2D(size=2,interpolation="bilinear")
        self.conv = tf.keras.Sequential(
            [
            DoubleConv(in_channels,residual=True),
            DoubleConv(out_channels,in_channels//2)
        ]
        )

        self.emb_layer = tf.keras.Sequential([
            SILU(),
            tf.keras.layers.Dense(out_channels)
        ]
        )
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def repeat(self,t,x):
        t = t[:,None,None,:]

        t = tf.repeat(t,[x.shape[1]],axis=1)
        t = tf.repeat(t,[x.shape[2]],axis=2)
        return t

    def call(self, x, skip_x,t):
        x = self.up(x)
        x = self.concat([skip_x,x])
        x = self.conv(x)
        t = self.emb_layer(t)
        emb = self.repeat(t,x)
        return x + emb


class Encoding(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoding, self).__init__()

    def call(self, t, channels):

        inv_freq = 1.0 / (10000 ** (tf.experimental.numpy.arange(start=0, stop=channels, step=2) / channels))
        t = tf.cast(t[..., tf.newaxis], dtype=tf.double)

        pos_enc_a = tf.math.sin(inv_freq * tf.repeat(t, repeats=[channels // 2], axis=-1))
        pos_enc_b = tf.math.cos(inv_freq * tf.repeat(t, repeats=[channels // 2], axis=-1))

        pos_enc = tf.concat([pos_enc_a, pos_enc_b], axis=-1)
        return pos_enc


class UNet(tf.keras.Model):
    def __init__(self,c_out=1, time_dim=64, **kwargs):
        super(UNet, self).__init__()

        self.time_dim = time_dim
        self.inc = DoubleConv(64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)

        self.bot1 = DoubleConv(512)
        self.bot2 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=512)
        self.bot3 = DoubleConv(256)

        self.up1 = Up(512, 128)
        self.sa3 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa4 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.outc = tf.keras.layers.Conv2D(c_out, kernel_size=1)
        self.pos_encoding = Encoding()

    def call(self, x, t):
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.inc(x)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4, x4, x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa3(x)
        x = self.up2(x, x2, t)
        x = self.sa4(x)
        x = self.up3(x, x1, t)
        output = self.outc(x)
        return output

if __name__ == '__main__':
    net = UNet()
    x = tf.experimental.numpy.random.randn(1,32,32,1)
    t = tf.constant([500] * x.shape[0])
    print(net(x, t).shape)
    net.summary()





