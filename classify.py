import numpy as np
import argparse
import tensorflow.compat.v2 as tf
from utils import IMG_SIZE, LABELS, image_generator


def classify(model, test_dir):
    """
    Classifies all images in test_dir
    :param model: Model to be evaluated
    :param test_dir: Directory including the images
    :return: None
    """
    test_img_gen = image_generator.flow_from_directory(test_dir,
                                                       target_size=(IMG_SIZE, IMG_SIZE),
                                                       classes=LABELS,
                                                       batch_size=1,
                                                       shuffle=False)

    ######### Your code starts here #########
    # Classify all images in the given folder
    # Calculate the accuracy and the number of test samples in the folder
    # test_img_gen has a list attribute filenames where you can access the filename of the datapoint
    num_test = len(test_img_gen.filenames)
    num_correct = 0
    for i in range(num_test):
        # extract next image
        x_i, y_i = next(test_img_gen)

        # do an evaluation with the model
        loss, acc = model.evaluate(x_i, y_i, verbose=0)

        # check if correct
        if acc:
            num_correct += 1
        else:
            print(test_img_gen.filenames[i])

    # calculate final accuracy
    accuracy = num_correct / num_test

    ######### Your code ends here #########

    print(f"Evaluated on {num_test} samples.")
    print(f"Accuracy: {accuracy*100:.0f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_image_dir', type=str, default='datasets/test/')
    FLAGS, _ = parser.parse_known_args()
    model = tf.keras.models.load_model('trained_models/trained.h5')
    classify(model, FLAGS.test_image_dir)