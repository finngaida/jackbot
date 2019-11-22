import matplotlib.pyplot as plt
import cv2 as cv
import argparse

import cards_finder
import detectRegion
import classifier


def test(show_debug=False):
    # load the ground truth labels globally once
    ranks, suits = classifier.load_dict()

    # these will store a series of bools indicating an (in)correct classification for later accuracy evaluation
    rank_history = []
    suit_history = []
    
    # test labels for the test images
    rank_labels = ['2', '9', '6', '7', '5', '5', 'dame', '4', '2', '8', '9', '10', 'roi', '1', '2', 'valet', 'dame', '3', '4', '9', '10', 'valet', 'roi', '1']
    suit_labels = ['diamonds', 'diamonds', 'clubs', 'clubs', 'hearts', 'clubs', 'diamonds', 'clubs', 'clubs', 'hearts', 'hearts', 'hearts', 'hearts', 'clubs', 'clubs', 'hearts', 'hearts', 'clubs', 'clubs', 'spades', 'spades', 'spades', 'spades', 'diamonds']

    for id in range(24):
        # 1. simulate card localization/morphing (tbd)
        img = cv.imread('test/{}.jpg'.format(id))
        binimg = classifier.binarize2(img)

        # 2. extract region of rank and suit from input
        rank_rect, suit_rect = detectRegion.detectRegion(img)

        # 4. predict the label
        r_result, s_result = classifier.classify(binimg, rank_rect, suit_rect, ranks, suits, show_debug)
        rank = r_result[0]
        suit = s_result[0]

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

    # calculate accuracy
    num_rank_successes = len([x for x in rank_history if x == True])
    num_rank_errors = len([x for x in rank_history if x == False])
    num_suit_successes = len([x for x in suit_history if x == True])
    num_suit_errors = len([x for x in suit_history if x == False])
    accuracy = (num_rank_successes + num_suit_successes) / (len(rank_history) + len(suit_history))
    print("accuracy: {}".format(accuracy))


def main():
    # load the ground truth labels globally once
    ranks, suits = classifier.load_dict()

    for i in range(1, 2):
        # load test image
        # img = cv.imread('test/{}.png'.format(i))
        img = cv.imread('test/wasd.jpg')
        # img = cards_finder.resize_image(img, scale=0.5)

        # find cards
        cards = cards_finder.select_cards(img)

        for card in cards:
            img = classifier.binarize2(card)
            # cards_finder.shi(img)

            # detect value region
            rank_rect, suit_rect = detectRegion.detectRegion(card)

            # classify

            result = classifier.classify(img, rank_rect, suit_rect, ranks, suits)

            if type(result) is str:
                print(result)
                plt.title(result)
            else:
                rank = result[0][0]
                suit = result[1][0]
                plt.title('{} of {}'.format(rank, suit))

            plt.imshow(cv.cvtColor(card, cv.COLOR_BGR2RGB))
            plt.show()

if __name__ == '__main__':
    main()
    # test()
