#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:03:08 2019

@author: finngaida
"""

import os
import matplotlib.pyplot as plt
import cv2 as cv
import argparse

import cards_detector
import region_detector
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
        img = cv.imread('../images/test/{}.jpg'.format(id))

        # 2. extract region of rank and suit from input
        rank_rect, suit_rect = region_detector.detect_region(img)

        # 4. predict the label
        result = classifier.classify(img, rank_rect, suit_rect, ranks, suits, show_debug)
        rank = result[0][0]
        suit = result[1][0]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input-path", help="Path to the cards scene image", required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.input_path):
        exit(-1)

    # load the ground truth labels globally once
    ranks, suits = classifier.load_dict()

    # load test image
    img = cv.imread(args.input_path)
    plt.title('Input image')
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

    # find cards
    print("Detecting cards in input image...")
    cards = cards_detector.detect_cards(img)
    print("Found {} cards:".format(len(cards)))

    for cardi in range(len(cards)):
        card = cards[cardi] # b 🙈

        # detect value region
        rank_rect, suit_rect = region_detector.detect_region(card)

        # classify
        result = classifier.classify(card, rank_rect, suit_rect, ranks, suits)
        plt.subplot(2, len(cards)//2+1, cardi+1)

        if type(result) is str:
            print(result)
            plt.title(result)
        else:
            rank = result[0][0]
            suit = result[1][0]
            card_title = '{} of {}'.format(rank, suit)
            plt.title(card_title)
            print(card_title)

        plt.imshow(cv.cvtColor(card, cv.COLOR_BGR2RGB))

    plt.show()

if __name__ == '__main__':
    main()
    # test()