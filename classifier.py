from skimage import io as skio
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import cv2 as cv

def binarize(img):
    """
    convert image to grayscale and then threshold it to become a binary map
    :param img: input image prefrably from skio
    :return: thresholded binary image
    """
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (3, 3), 0)
    ret, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    return image

def binarize2(img):
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (3, 3), 0)
    ret, image = cv.threshold(image, 200, 255, cv.THRESH_BINARY_INV)
    return image

def load_dict(ranks_to_load = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'valet', 'dame', 'roi'],
              suits_to_load = ['hearts', 'diamonds', 'clubs', 'spades']):
    """
    generate the "ground truth" labelled datasets to compare against
    :param ranks_to_load: default: all
    :param suits_to_load: default: all
    :return: ranks, suits dictionaries with labels as keys and images as values
    """

    ranks = {}
    suits = {}
    for rank in ranks_to_load:
        img = binarize(skio.imread('groundtruth/ranks/{}.jpg'.format(rank)))
        ranks[rank] = img
    for suit in suits_to_load:
        img = binarize(skio.imread('groundtruth/suits/{}.jpg'.format(suit)))
        suits[suit] = img
    return ranks, suits

def class_probabilities(image, dict, show_debug=False):
    """
    calculate the probabilities of the input image belonging to any of the specified classes in dict
    :param image: input image (preferably binarized)
    :param dict: class labels to check against (load with load_dict)
    :param show_debug: set to true to render images and diffs
    :return: dict with labels as keys and corresponding probabilities as values
    """
    probabilities = {}
    for label, compare_img in dict.items():
        # equalize size
        ch = compare_img.shape[0]
        cw = compare_img.shape[1]
        img = resize(image, (ch, cw), preserve_range=True)

        # take diff
        diff = np.abs(img - compare_img)
        histo, _ = np.histogram(diff, bins=20)
        histo = histo[1:]
        probabilities[label] = sum(histo)

        # show images/diff
        if show_debug:
            plt.subplot(321)
            plt.title('orig')
            plt.imshow(image, cmap='gray')
            plt.subplot(322)
            plt.title('compare_img')
            plt.imshow(compare_img, cmap='gray')
            plt.subplot(323)
            plt.title('diff')
            plt.imshow(diff, cmap='gray')
            plt.show()

    return probabilities

def prediction_tuple(probabilities):
    """
    Extract the most probable prediction from the dict of all predictions
    :param probabilities: all predictions
    :return: most probable label and accompanying confidence
    """
    prediction = [x for x, i in probabilities.items() if i == min(probabilities.values())][0]
    return (prediction, probabilities[prediction])

def classify(image, rank_rect, suit_rect, rank_labels, suit_labels, show_debug=False):
    """
    Classify an image into one of the supplied classes
    :param image: input image (preferably binarized)
    :param labels: labels dict (load with load_dict)
    :return: predicted label and dict of all probabilities
    """

    # abort if no region was found
    if rank_rect[2] == 0 or rank_rect[3] == 0 or suit_rect[2] == 0 or suit_rect[3] == 0:
        return "Insufficient regions: {}, {}".format(rank_rect, suit_rect)

    # 3. crop image to the rank/suit region
    img = binarize2(image)
    rank_img = img[rank_rect[1]:rank_rect[1] + rank_rect[3], rank_rect[0]:rank_rect[0] + rank_rect[2]]
    suit_img = img[suit_rect[1]:suit_rect[1] + suit_rect[3], suit_rect[0]:suit_rect[0] + suit_rect[2]]

    # calculate probabilities for each class
    rank_probabilities = class_probabilities(rank_img, rank_labels)
    suit_probabilities = class_probabilities(suit_img, suit_labels)

    # extract prediction
    r_pred, _ = prediction_tuple(rank_probabilities)
    s_pred, _ = prediction_tuple(suit_probabilities)

    if show_debug:
        plt.subplot(221)
        plt.imshow(rank_img, cmap='gray')
        plt.subplot(222)
        buckets = np.arange(len(rank_probabilities))
        plt.bar(buckets, rank_probabilities.values())
        plt.xticks(buckets, rank_probabilities.keys())
        plt.subplot(223)
        plt.imshow(suit_img, cmap='gray')
        plt.title("{} of {}".format(r_pred, s_pred))
        plt.subplot(224)
        buckets = np.arange(len(suit_probabilities))
        plt.bar(buckets, suit_probabilities.values())
        plt.xticks(buckets, suit_probabilities.keys())
        plt.show()

    return (r_pred, rank_probabilities), (s_pred, suit_probabilities)