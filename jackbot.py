from skimage import io as skio
import numpy as np
from skimage.measure import _structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt
import sklearn.neighbors as skneigh
import cv2 as cv
import detectRegion

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
        histo, _ = np.histogram(diff, bins=bins)
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

def classify(image, labels):
    """
    Classify an image into one of the supplied classes
    :param image: input image (preferably binarized)
    :param labels: labels dict (load with load_dict)
    :return: predicted label and dict of all probabilities
    """

    # calculate probabilities for each class
    probabilities = class_probabilities(image, labels)

    # extract prediction
    pred, prob = prediction_tuple(probabilities)

    return pred, probabilities


def main(show_debug=False):
    # load the ground truth labels globally once
    ranks, suits = load_dict()

    # these will store a series of bools indicating an (in)correct classification for later accuracy evaluation
    rank_history = []
    suit_history = []

    # test labels for the test images
    rank_labels = ['2', '9', '6', '7', '5', '5', 'dame', '4', '2', '8', '9', '10', 'roi', '1', '2', 'valet', 'dame', '3', '4', '9', '10', 'valet', 'roi', '1']
    suit_labels = ['diamonds', 'diamonds', 'clubs', 'clubs', 'hearts', 'clubs', 'diamonds', 'clubs', 'clubs', 'hearts', 'hearts', 'hearts', 'hearts', 'clubs', 'clubs', 'hearts', 'hearts', 'clubs', 'clubs', 'spades', 'spades', 'spades', 'spades', 'diamonds']

    for id in range(24):
        # 1. simulate card localization/morphing (tbd)
        img = binarize(skio.imread('test/{}.jpg'.format(id)))

        # 2. extract region of rank and suit from input
        rank_rect, suit_rect = detectRegion.detectRegion(img)

        # abort if no region was found
        if rank_rect[2] == 0 or rank_rect[3] == 0 or suit_rect[2] == 0 or suit_rect[3] == 0:
            print("Couldn't find value region for {}".format(id))
            rank_history.append(False)
            suit_history.append(False)
            continue

        # 3. crop image to the rank/suit region
        rank_img = img[rank_rect[1]:rank_rect[1]+rank_rect[3], rank_rect[0]:rank_rect[0]+rank_rect[2]]
        suit_img = img[suit_rect[1]:suit_rect[1] + suit_rect[3], suit_rect[0]:suit_rect[0] + suit_rect[2]]

        # 4. predict the label
        rank, r_prob = classify(rank_img, ranks)
        suit, s_prob = classify(suit_img, suits)

        # evaluation logic
        rank_success = rank == rank_labels[id]
        rank_history.append(rank_success)
        suit_success = suit == suit_labels[id]
        suit_history.append(suit_success)

        print("{} {} {}, {} {} {} {}".format(
            rank,
            "==" if rank_success else "!=",
            rank_labels[id],
            suit,
            "==" if suit_success else "!=",
            suit_labels[id],
            "✅" if rank_success and suit_success else "🔴"))

        if show_debug:
            plt.subplot(221)
            plt.imshow(rank_img, cmap='gray')
            plt.subplot(222)
            buckets = np.arange(len(r_prob))
            plt.bar(buckets, r_prob.values())
            plt.xticks(buckets, r_prob.keys())
            plt.subplot(223)
            plt.imshow(suit_img, cmap='gray')
            plt.title("{} of {}".format(rank, suit))
            plt.subplot(224)
            buckets = np.arange(len(s_prob))
            plt.bar(buckets, s_prob.values())
            plt.xticks(buckets, s_prob.keys())
            plt.show()

    # calculate accuracy
    num_rank_successes = len([x for x in rank_history if x == True])
    num_rank_errors = len([x for x in rank_history if x == False])
    num_suit_successes = len([x for x in suit_history if x == True])
    num_suit_errors = len([x for x in suit_history if x == False])
    accuracy = (num_rank_successes + num_suit_successes) / (len(rank_history) + len(suit_history))
    print("accuracy: {}".format(accuracy))


if __name__ == '__main__':
    main()