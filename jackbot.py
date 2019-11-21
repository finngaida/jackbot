from skimage import io as skio
import numpy as np
from skimage.measure import _structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt
import sklearn.neighbors as skneigh
import cv2 as cv
import detectRegion

rank_to_suit_ratio = 0.56

def binarize(img):
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (3, 3), 0)
    ret, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    return image

def load_dict(ranks_to_load = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'valet', 'dame', 'roi'],
              suits_to_load = ['hearts', 'diamonds', 'clubs', 'spades']):
    """

    :param ranks_to_load:
    :param suits_to_load:
    :return:
    """

    ranks = {}
    suits = {}
    for rank in ranks_to_load:
        img = binarize(skio.imread('groundtruth/ranks/{}.jpg'.format(rank)))
        # histo, _ = np.histogram(img, bins=2)
        ranks[rank] = img#, histo)
    for suit in suits_to_load:
        img = binarize(skio.imread('groundtruth/suits/{}.jpg'.format(suit)))
        # histo, _ = np.histogram(img, bins=2)
        suits[suit] = img#, histo)
    return ranks, suits

def class_probabilities(image, dict, bins=2, show_debug=False):
    """

    :param image:
    :param dict:
    :param bins:
    :param show_debug:
    :return:
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

    :param probabilities:
    :return:
    """
    prediction = [x for x, i in probabilities.items() if i == min(probabilities.values())][0]
    return (prediction, probabilities[prediction])

def classify(image, labels):
    """

    :param image:
    :param labels:
    :return:
    """

    # calculate probabilities for each class
    probabilities = class_probabilities(image, labels)

    # extract prediction
    pred, prob = prediction_tuple(probabilities)

    return pred, probabilities

def test_accuracy():
    rank_history = []
    suit_history = []
    ranks, suits = load_dict()
    for card in range(1, 9):
            img_ = skio.imread('test/{}.jpg'.format(card))
            img = skio.imread('test/{}.jpg'.format(card), as_gray=True)
            rank_pred, suit_pred = classify(img, ranks, suits)
            rank_success = rank_pred[0] == card[0]
            rank_history.append(rank_success)
            suit_success = suit_pred[0] == card[1]
            suit_history.append(suit_success)

            print("{} {} {}, {} {} {} {}".format(
                rank_pred,
                "==" if rank_success else "!=",
                card[0],
                suit_pred,
                "==" if suit_success else "!=",
                card[1],
                "✅" if rank_success and suit_success else "🔴"))

    num_rank_successes = len([x for x in rank_history if x == True])
    num_rank_errors = len([x for x in rank_history if x == False])
    num_suit_successes = len([x for x in suit_history if x == True])
    num_suit_errors = len([x for x in suit_history if x == False])
    accuracy = (num_rank_successes + num_suit_successes) / (len(rank_history) + len(suit_history))
    print("accuracy: {}".format(accuracy))

def main(show_debug=False):
    ranks, suits = load_dict()
    rank_history = []
    suit_history = []

    rank_labels = ['2', '9', '6', '7', '5', '5', 'dame', '4', '2', '8', '9', '10', 'roi', '1', '2', 'valet', 'dame', '3', '4', '9', '10', 'valet', 'roi', '1']
    suit_labels = ['diamonds', 'diamonds', 'clubs', 'clubs', 'hearts', 'clubs', 'diamonds', 'clubs', 'clubs', 'hearts', 'hearts', 'hearts', 'hearts', 'clubs', 'clubs', 'hearts', 'hearts', 'clubs', 'clubs', 'spades', 'spades', 'spades', 'spades', 'diamonds']

    for id in range(24):
        # 1. simulate card localization/morphing (tbd)
        img = binarize(skio.imread('test/{}.jpg'.format(id)))

        # 2. extract region of rank and suit from input
        rank_rect, suit_rect = detectRegion.detectRegion(img)

        if rank_rect[2] == 0 or rank_rect[3] == 0 or suit_rect[2] == 0 or suit_rect[3] == 0:
            print("Couldn't find value region for {}".format(id))
            rank_history.append(False)
            suit_history.append(False)
            continue

        # 3. classify
        rank_img = img[rank_rect[1]:rank_rect[1]+rank_rect[3], rank_rect[0]:rank_rect[0]+rank_rect[2]]
        suit_img = img[suit_rect[1]:suit_rect[1] + suit_rect[3], suit_rect[0]:suit_rect[0] + suit_rect[2]]

        rank, r_prob = classify(rank_img, ranks)
        suit, s_prob = classify(suit_img, suits)

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

    num_rank_successes = len([x for x in rank_history if x == True])
    num_rank_errors = len([x for x in rank_history if x == False])
    num_suit_successes = len([x for x in suit_history if x == True])
    num_suit_errors = len([x for x in suit_history if x == False])
    accuracy = (num_rank_successes + num_suit_successes) / (len(rank_history) + len(suit_history))
    print("accuracy: {}".format(accuracy))


if __name__ == '__main__':
    main()