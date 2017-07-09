# -------------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------------
import keras
from keras.datasets import mnist
import keras.backend as K
import matplotlib.pyplot as plt

import numpy as np

MNIST_NUM_CLASSES = 10


def deprocess_image(x):
    """
    Utility function to convert a tensor into a valid image
    Normalize Tensor: Center on 0., ensure std is 0.1 [Why?]

    REF: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py.
    Unlike the original ref, data is kept within the [0,1] range. Matplotlib.pyplot.imshow prefers
    values in this range rather than [0, 255] which scipy.misc.imshow prefers. The later hangs
     the code until the figure is closed.

    :param x: image of dimension [r, c, ch]
    :return:
    """
    x -= x.mean()
    x /= (x.std() + 1e-5)

    # Clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    print("inside deprocess_image")

    # if K.image_data_format() == 'channels_first':  # [ch,r, c]
    #     x = x.transpose((1, 2, 0))  # this is similar to K.permute dimensions but outside keras/TF

    return x


def get_mnist_data(sample_idx=0):
    """
    :param sample_idx: index of sample to return

    :return: data_train, label_train, data_test, label_test, data_sample, label_sample
    """
    (data_train, label_train), (data_test, label_test) = mnist.load_data()
    data_train = data_train.reshape(data_train.shape[0], data_train.shape[1], data_train.shape[2], 1)
    data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], data_train.shape[2], 1)

    data_train = data_train.astype('float32')
    data_test = data_test.astype('float32')

    data_train /= 255
    data_test /= 255

    label_train = keras.utils.to_categorical(label_train, MNIST_NUM_CLASSES)
    label_test = keras.utils.to_categorical(label_test, MNIST_NUM_CLASSES)

    # a Single sample to test prediction on
    data_sample = data_test[sample_idx].reshape(
        1, data_test[sample_idx].shape[0], data_test[sample_idx].shape[1], data_test[sample_idx].shape[2])
    label_sample = label_test[sample_idx]

    return data_train, label_train, data_test, label_test, data_sample, label_sample


def display_filters(weights, margin=1):
    """
    Display the filters of a layer in a single large image

    :param weights: weight matrix of a model layer
    :param margin: Gap/border between the kernels, filters in the large tiled filters image.

    :return:
    """

    if len(K.int_shape(weights)) == 3:
        r, c, out_ch = K.int_shape(weights)
        weights = K.reshape(weights, (r, c, 1, out_ch))
    r, c, in_ch, out_ch = K.int_shape(weights)

    allowed_in_ch = [1, 3]  # can only display filters where the input dimension is 1 or 3
    if in_ch not in allowed_in_ch:
        raise Exception("Cannot display filters with input channels = %d" % in_ch)

    n = np.int(np.round(np.sqrt(out_ch)))  # Single dimension of tiled image

    width = (n * r) + ((n - 1) * margin)
    height = (n * c) + ((n - 1) * margin)

    tiled_filters = np.zeros((width, height, in_ch))
    print(tiled_filters.shape)

    # Fill in in composite image with the filters
    for r_idx in range(n):
        for c_idx in range(n):

            filt_idx = (r_idx * n) + c_idx
            if filt_idx >= out_ch:
                break

            print("Processing filter %d" % filt_idx)

            tiled_filters[
                (r + margin) * r_idx: (r + margin) * r_idx + r,
                (c + margin) * c_idx: (c + margin) * c_idx + c,
                :
            ] = deprocess_image(K.eval(weights[:, :, :, filt_idx]))

    # Plot the Composite Figure
    plt.ion()
    plt.figure()

    if 1 == in_ch:
        plt.imshow(tiled_filters[:, :, 0], cmap='Greys')  # force to 2D. Expected by imshow
    else:
        plt.imshow(tiled_filters)
    plt.colorbar()

    return tiled_filters


def display_layer_activations(model, layer_idx, data_sample):
    """
    Display the activation volume of the specified layer. Each feature map is displayed in a separate subplot.
    Expected format of layers is [b, r, c, ch]

    :param model:
    :param layer_idx:
    :param data_sample:

    :return: the whole activation volume
    """

    # Define a function to get the activation volume
    get_layer_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_idx].output]
    )

    # Get the activations in a usable format
    act_volume = np.asarray(get_layer_output(
        [data_sample, 0],  # second input specifies the learning phase 0=output, 1=training
    ))

    # Reshape the activations, the casting above adds another dimension
    act_volume = act_volume.reshape(
        act_volume.shape[1],
        act_volume.shape[2],
        act_volume.shape[3],
        act_volume.shape[4]
    )

    max_ch = np.int(np.round(np.sqrt(act_volume.shape[-1])))

    f = plt.figure()

    for ch_idx in range(act_volume.shape[-1]):
        f.add_subplot(max_ch, max_ch, ch_idx + 1)
        plt.imshow(act_volume[0, :, :, ch_idx], cmap='Greys')

    f.suptitle("Feature maps of layer @ idx %d: %s" % (layer_idx, model.layers[layer_idx].name))

    return act_volume


def add_noise(images, noise_type, **kwargs):
    """

    :param images:
    :param noise_type: ['gaussian', 'pepper']
    :param kwargs: conditional keyword arguments

    if noise_type = gaussian
    var = noise variance. default = 0.1
    mean = mean of the noise. default = 0

    :return: noisy images

    REF: https://stackoverflow.com/questions/22937589/
         how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """

    noise_type = noise_type.lower()
    allowed_noise = ['gaussian', 'pepper']

    if noise_type not in allowed_noise:
        raise Exception("Unknown noise type, %s" % noise_type)

    b, r, c, ch = images.shape

    if noise_type == 'gaussian':

        var = 0.1
        mean = 0

        if 'var' in kwargs.keys():
            var = kwargs['var']
        if 'mean' in kwargs.keys():
            mean = kwargs['mean']

        print("Adding Gaussian noise with mean %f, var %f to images of size %s" % (mean, var, images.shape))

        sigma = np.sqrt(var)
        noise = np.random.normal(mean, sigma, (b, r, c, ch))

        output = images + noise

    elif noise_type == 'pepper':
        prob = 0.004

        if 'prob' in kwargs.keys():
            prob = kwargs['prob']

        # num samples to blacken
        num_pepper = int(prob * (b * r * c * ch))

        print("Adding Pepper noise with probability %f to images of size %s" % (prob, images.shape))

        xs = [np.random.randint(0, max(i-1, 1), int(num_pepper)) for i in images.shape]
        xs = np.array(xs)

        # force numpy to create a separate copy of the input
        output = np.copy(images)

        output[xs[0, :], xs[1, :], xs[2,:], xs[3, :]] = 0

    return output
