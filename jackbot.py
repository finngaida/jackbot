from skimage import io as skio
import numpy as np
from skimage.measure import _structural_similarity as ssim
from skimage.transform import resize
import matplotlib.pyplot as plt

def load_dict(cards_to_load = ['1h', '1d', '1c', '1s', 'rh', 'rd', 'rc', 'rs', '2d', '5s', '6d', 'ds']):
    dict = {}
    for card in cards_to_load:
        dict[card] = skio.imread('groundtruth/{}.jpeg'.format(card), as_gray=True)
    return dict

def classify(image, dict, bins=20):
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

        if False:
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

    prediction = [x for x, i in probabilities.items() if i == min(probabilities.values())][0]
    return (prediction, probabilities[prediction])


dict = load_dict()
history = []
for card in ['1c', '1d', '1h', '1s', 'rc', 'rd', 'rh', '5s']:
    for i in range(2, 4):
        prediction, probability = classify(skio.imread('asdfa/{}{}.jpeg'.format(card, i), as_gray=True), dict)
        success = prediction == card
        history.append(success)
        print("{} {} {} {}".format(prediction, "==" if success else "!=", card, "✅" if success else "🔴"))

num_successes = len([x for x in history if x == True])
num_errors = len([x for x in history if x == False])
accuracy = num_successes / len(history)
print("accuracy: {}".format(accuracy))