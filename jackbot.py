from skimage import io as skio
import numpy as np
from skimage.measure import _structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt

rank_to_suit_ratio = 0.56

def load_dict(ranks_to_load = ['1', '2', '5', '6', 'dame', 'roi'], suits_to_load = ['hearts', 'diamonds', 'clubs', 'spades']):
    ranks = {}
    suits = {}
    for rank in ranks_to_load:
        ranks[rank] = skio.imread('groundtruth/ranks/{}.jpg'.format(rank), as_gray=True)
    for suit in suits_to_load:
        suits[suit] = skio.imread('groundtruth/suits/{}.jpg'.format(suit), as_gray=True)
    return ranks, suits

def class_probabilities(image, dict, bins=20, show_debug=False):
    probabilities = {}
    for label, compare in dict.items():
        # equalize size
        ch = compare.shape[0]
        cw = compare.shape[1]
        img = resize(image, (ch, cw))

        # take diff
        diff = np.abs(img - compare)
        histo, _ = np.histogram(diff, bins=bins)
        histo = histo[bins//4:]
        probabilities[label] = sum(histo)

        # show images/diff
        if show_debug:
            plt.subplot(221)
            plt.title('orig')
            plt.imshow(image, cmap='gray')
            plt.subplot(222)
            plt.title('compare')
            plt.imshow(compare, cmap='gray')
            plt.subplot(223)
            plt.title('diff')
            plt.imshow(diff, cmap='gray')
            plt.subplot(224)
            plt.title('histo')
            plt.plot(range(bins//4, bins), histo)
            plt.show()

    return probabilities

def prediction_tuple(probabilities):
    prediction = [x for x, i in probabilities.items() if i == min(probabilities.values())][0]
    return (prediction, probabilities[prediction])

def classify(image, ranks, suits):
    # separate input image into rank and suit part
    height = image.shape[0]
    cutoff = int(height * rank_to_suit_ratio)
    rank_part = image[0:cutoff][:]
    suit_part = image[cutoff:height][:]

    # calculate probabilities for each class
    rank_probabilities = class_probabilities(rank_part, ranks)
    suit_probabilities = class_probabilities(suit_part, suits)

    # extract prediction
    rank_pred, rank_prob = prediction_tuple(rank_probabilities)
    suit_pred, suit_prob = prediction_tuple(suit_probabilities)

    return rank_pred, suit_pred

def main():
    rank_history = []
    suit_history = []
    ranks, suits = load_dict()
    for card in ['1c', '1d', '1h', '1s', '5s', 'rc', 'rd', 'rh']:
        for i in range(2, 4):
            img = skio.imread('test/{}{}.jpeg'.format(card, i), as_gray=True)
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

if __name__ == '__main__':
    main()