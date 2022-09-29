import tensorflow as tf
from config import *
import time

@tf.function
def train_step(batch):
    timestep_values = generate_timestamp(batch.shape[0])
    noised_image, noise = forward_noise(batch, timestep_values)
    with tf.GradientTape() as tape:
        prediction = net(noised_image, timestep_values)

        loss_value = loss_fn(noise, prediction)

    gradients = tape.gradient(loss_value, net.trainable_variables)
    opt.apply_gradients(zip(gradients, net.trainable_variables))
    train_acc_metric.update_state(noise, prediction)
    return loss_value


@tf.function
def test_step(batch):
    timestep_values = generate_timestamp(batch.shape[0])

    noised_image, noise = forward_noise(batch, timestep_values)

    prediction = net(noised_image, timestep_values)
    loss_value = loss_fn(noise, prediction)
    # Update training metric.
    val_acc_metric.update_state(noise, prediction)
    return loss_value

def main():
    for e in range(num_epochs):
        print("\nStart of epoch %d" % (e,))
        start_time = time.time()

        # this is cool utility in Tensorflow that will create a nice looking progress bar
        for i, batch in enumerate(iter(train_generator)):
            # run the training loop
            loss = train_step(batch)

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()

        print("Training MSE: %.4f" % (float(train_acc),))
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        for i, batch in enumerate(iter(val_generator)):
            # run the training loop
            val_loss = test_step(batch)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()

        print("Validation MSE: %.4f" % (float(val_acc),))
        # print("validation KID: %.4f" % (float(val_kid),))
        print("Time taken: %.2fs" % (time.time() - start_time))

        ckpt_manager.save(checkpoint_number=e)

if __name__ == '__main__':
    main()