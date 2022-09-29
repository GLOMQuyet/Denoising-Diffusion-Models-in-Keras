import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_paths, image_size, batch_size, dim=(28, 28), n_channels=1, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.image_size = [image_size, image_size]
        self.img_paths = img_paths
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.img_indexes = np.arange(len(self.img_paths))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temps)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temps):
        X = np.empty((self.batch_size, *self.dim))

        for i, ID in enumerate(list_IDs_temps):
            X[i,] = self.img_paths[ID]
        X = X[:, :, :, np.newaxis]
        X = tf.image.resize(X, size=self.image_size)
        X = tf.cast(X / 255.0, dtype=tf.float32)
        return X