import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf1
import tensorflow.compat.v2 as tf

from utils import decode_jpeg, IMG_SIZE, normalize_resize_image, LABELS, maybe_makedirs

tf1.compat.v1.enable_eager_execution()


def compute_brute_force_classification(model, image_path, nH=8, nW=8):
    '''
    This function returns the probabilities of each window.
    Inputs:
        model: Model which is used
        image_path: path to the image to be analysed
        nH: number of windows in the vertical direction
        nW: number of windows in the horizontal direction
    Outputs:
        window_predictions: a (nH, nW, 3) np.array.
                            The last dim (size 3) is the probabilities
                            of each label (cat, dog, neg)
    HINT: normalize_resize_image  (from utils.py) will be useful here.
    HINT: If you want to predict a single image you have to add a singular batch dimension:
            [IMG_SIZE, IMG_SIZE, 3] -> [1, IMG_SIZE, IMG_SIZE, 3].
            Similarly predict will return a [1, 3] array which you might want to squeeze into a [3] array
    '''

    # H x W x 3 numpy array (3 for each RGB color channel)
    raw_image = decode_jpeg(image_path).numpy()

    ######### Your code starts here #########
    # calculate the height and width of the sliding window without padding
    no_pad_window_height = int(round(raw_image.shape[0] / nH))
    no_pad_window_width = int(round(raw_image.shape[1] / nW))

    # add padding to the raw_image
    horiz_pad_size = no_pad_window_width
    vert_pad_size = no_pad_window_height
    padded_image = np.pad(raw_image, ((vert_pad_size, vert_pad_size), (horiz_pad_size, horiz_pad_size), (0, 0)))

    # initialize the empty window predictions array
    window_predictions = np.zeros((nH, nW, 3))

    # create the nW and nH pairings in a list - pre-computing this to avoid nested for loops
    nW_nH_pairings = [(i, j) for i in range(nH) for j in range(nW)]

    # for each sub-image
    for curr_nH, curr_nW in nW_nH_pairings:
        # compute the coordinates of the current window
        x_start = curr_nW * no_pad_window_width
        x_end = (curr_nW + 1) * no_pad_window_width + 2 * horiz_pad_size
        y_start = curr_nH * no_pad_window_height
        y_end = (curr_nH + 1) * no_pad_window_height + 2 * vert_pad_size

        # grab the sub-image, normalize and resize
        curr_window = normalize_resize_image(padded_image[y_start:y_end, x_start:x_end, :], IMG_SIZE)

        # add singular batch dimension
        expanded_curr_window = np.expand_dims(curr_window, axis=0)

        # predict with the model for the image and squeeze into a [3] array
        prediction = np.squeeze(model.predict(expanded_curr_window))

        # append the predictions to the proper index of the output
        window_predictions[curr_nH, curr_nW, :] = prediction

    ######### Your code ends here #########

    return window_predictions


def compute_convolutional_KxK_classification(model, image_path):
    """
    Computes probabilities for each window based on the convolution layer of Inception
    :param model:Model which is used
    :param image_path: Path to the image to be analysed
    :return: None
    """
    raw_image = decode_jpeg(image_path).numpy()
    resized_patch = normalize_resize_image(raw_image, IMG_SIZE)
    conv_model = tf.keras.Model(model.layers[0].inputs, model.layers[0].layers[-2].output)

    ######### Your code starts here #########
    # We want to use the output of the last convolution layer which has the shape [bs, K, K, bottleneck_size]
    # expand the dimensions of the resized_patch
    resized_patch = np.expand_dims(resized_patch, axis=0)

    # get the convolution layer output
    conv_output = conv_model.predict(resized_patch)

    # get the value of K
    K = conv_output.shape[1]

    # reshape the convolution output
    reshaped_conv_output = np.reshape(conv_output, [K*K, conv_output.shape[3]])

    # run the reshaped output through our linear classifier to get each KxK patch's prediction
    linear_classifier_model = tf.keras.Sequential(model.get_layer('classifier'))
    predictionsKxK = linear_classifier_model.predict(reshaped_conv_output)

    ######### Your code ends here #########

    return np.reshape(predictionsKxK, [K, K, -1])


def compute_and_plot_saliency(model, image_path):
    """
    This function computes and plots the saliency plot.
    You need to compute the matrix M detailed in section 3.1 in
    K. Simonyan, A. Vedaldi, and A. Zisserman,
    "Deep inside convolutional networks: Visualising imageclassification models and saliency maps,"
    2013, Available at https://arxiv.org/abs/1312.6034.

    :param model: Model which is used
    :param image_path: Path to the image to be analysed
    :return: None
    """
    raw_image = tf.dtypes.cast(decode_jpeg(image_path), tf.float32)

    logits_tensor = model.get_layer('classifier')
    logits_model = tf.keras.Model(model.input, logits_tensor.output)

    ######### Your code starts here #########

    with tf.GradientTape(persistent=True) as t:
        # watch the input variable (aka the expanded_image)
        t.watch(raw_image)

        # first process the raw image
        processed_image = tf.expand_dims(normalize_resize_image(raw_image, IMG_SIZE), axis=0)

        # calculate the y label using the logits model
        y_pred = logits_model(processed_image)

        # squeeze y_pred to reduce dimension
        y_pred = tf.squeeze(y_pred)

        # find the index of the highest scored class so that we take the maximum gradient w
        top_class = tf.argmax(y_pred)
        top_y_pred = y_pred[top_class]

    # calculate the gradient w for the highest score at each position in the expanded image
    grads = t.gradient(top_y_pred, raw_image)

    # calculate the map by taking the max of each color channel
    M = np.max(grads, axis=2)
    ######### Your code ends here #########

    plt.subplot(2, 1, 1)
    plt.imshow(M)
    plt.title('Saliency with respect to predicted class %s' % LABELS[top_class])
    plt.subplot(2, 1, 2)
    plt.imshow(decode_jpeg(image_path).numpy())
    plt.savefig("../plots/saliency.png")
    plt.show()


def plot_classification(image_path, classification_array):
    nH, nW, _ = classification_array.shape
    image_data = decode_jpeg(image_path).numpy()
    aspect_ratio = float(image_data.shape[0]) / image_data.shape[1]
    plt.figure(figsize=(8, 8*aspect_ratio))
    p1 = plt.subplot(2,2,1)
    plt.imshow(classification_array[:,:,0], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[0])
    p1.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    p2 = plt.subplot(2,2,2)
    plt.imshow(classification_array[:,:,1], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[1])
    p2.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    p2 = plt.subplot(2,2,3)
    plt.imshow(classification_array[:,:,2], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[2])
    p2.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(image_data)
    plt.savefig("../plots/detect.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--scheme', type=str)
    FLAGS, _ = parser.parse_known_args()
    maybe_makedirs("../plots")

    model = tf.keras.models.load_model('trained_models/trained.h5')
    if FLAGS.scheme == 'brute':
        plot_classification(FLAGS.image, compute_brute_force_classification(model, FLAGS.image, 8, 8))
    elif FLAGS.scheme == 'conv':
        plot_classification(FLAGS.image, compute_convolutional_KxK_classification(model, FLAGS.image))
    elif FLAGS.scheme == 'saliency':
        compute_and_plot_saliency(model, FLAGS.image)
    else:
        print('Unrecognized scheme:', FLAGS.scheme)