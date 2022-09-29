import matplotlib.pyplot as plt
# Save a GIF using logged images
from PIL import Image
from config import *

def plot_forward():
    # Let us visualize the output image at a few timestamps
    sample_mnist = train_generator.__getitem__(1)[0]

    for index, i in enumerate([0,50,100,150,200,250,300,350,400,450,500]):
        noisy_im, noise = forward_noise(sample_mnist, np.array([i,]))
        plt.subplot(1, 11, index+1)
        plt.imshow(np.squeeze(noisy_im))
    plt.show()

def save_gif(img_list, path="", interval=500):
    # Transform images from [-1,1] to [0, 255]
    imgs = []
    for im in img_list:
        im = np.array(im)
        im = (im + 1) * 127.5
        im = np.clip(im, 0, 255).astype(np.int32)
        im = Image.fromarray(im)
        imgs.append(im)

    imgs = iter(imgs)

    # Extract first image from iterator
    img = next(imgs)

    # Append the other images and save as GIF
    img.save(fp=path, format='GIF', append_images=imgs,
             save_all=True, duration=interval, loop=0)

def plot_backward():
    x = tf.random.normal((1, 32, 32, 1))
    img_list = []
    img_list.append(np.squeeze(np.squeeze(x, 0), -1))
    for i in range(timesteps):
        t = np.expand_dims(np.array(timesteps - i, np.int32), 0)
        pred_noise = net(x, t)
        x = ddpm(x, pred_noise, t)

        img_list.append(np.squeeze(np.squeeze(x, 0), -1))
        if i % 100 == 0:
            img = np.squeeze(x[0])
            plt.imshow(np.array(np.clip((img + 1) * 127.5, 0, 255), np.uint8))
            plt.show()
    save_gif(img_list + ([img_list[-1]] * 100), "ddpm.gif", interval=20)
    plt.imshow(np.array(np.clip(img, a_min=0, a_max=255)))
    plt.show()
plot_backward()
