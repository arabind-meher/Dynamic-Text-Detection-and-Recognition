########################################################################################################################
# Filename: static_text_recognition.py
# Description: Recognizes text from image
# Usage: python static_text_recognition.py
#        or
#        python static_text_recognition.py --image images/001.png
#        (python static_text_recognition.py -i | -c | -w | -e | -p (arguments))
# Author: Arabind Meher (ag3774@srmist.edu.in) & Karan Gopalakrishnan (ck4659@srmist.edu.in)
########################################################################################################################

import cv2
import argparse
import pytesseract
import numpy as np
from imutils.object_detection import non_max_suppression

from utils import resize_image, forward_passer, box_extractor

# setting up tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# setting EAST module
east = 'frozen_east_text_detection.pb'


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', type=str,
                    help='path to image')
    ap.add_argument('-c', '--min_confidence', type=float, default=0.5,
                    help='minimum confidence to process a region')
    ap.add_argument('-w', '--width', type=int, default=320,
                    help='resized image width (multiple of 32)')
    ap.add_argument('-e', '--height', type=int, default=320,
                    help='resized image height (multiple of 32)')
    ap.add_argument('-p', '--padding', type=float, default=0.0,
                    help='padding on each ROI border')
    arguments = vars(ap.parse_args())

    return arguments


def main(image, width, height, min_confidence, padding):
    # reading in image
    image = cv2.imread(image)
    orig_image = image.copy()
    orig_h, orig_w = orig_image.shape[:2]

    # resizing image
    image, ratio_w, ratio_h = resize_image(image, width, height)

    # layers used for ROI recognition
    layer_names = ['feature_fusion/Conv_7/Sigmoid',
                   'feature_fusion/concat_3']

    # pre-loading the frozen graph
    print("[INFO] loading the detector...")
    net = cv2.dnn.readNet(east)

    # getting results from the model
    scores, geometry = forward_passer(net, image, layers=layer_names)

    # decoding results from the model
    rectangles, confidences = box_extractor(scores, geometry, min_confidence)

    # applying non-max suppression to get boxes depicting text regions
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    results = []

    # text recognition main loop
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)

        dx = int((end_x - start_x) * padding)
        dy = int((end_y - start_y) * padding)

        start_x = max(0, start_x - dx)
        start_y = max(0, start_y - dy)
        end_x = min(orig_w, end_x + (dx * 2))
        end_y = min(orig_h, end_y + (dy * 2))

        # ROI to be recognized
        roi = orig_image[start_y:end_y, start_x:end_x]

        # recognizing text
        config = '-l eng --oem 1 --psm 7'
        text = pytesseract.image_to_string(roi, config=config)

        # collating results
        results.append(((start_x, start_y, end_x, end_y), text))

    # sorting results top to bottom
    results.sort(key=lambda r: r[0][1])
    final_text = list()

    # printing OCR results & drawing them on the image
    for (start_x, start_y, end_x, end_y), text in results:
        print('OCR Result')
        print('**********')
        print(f'{text}\n')

        # stripping out ASCII characters
        text = ''.join([c if ord(c) < 128 else "" for c in text]).strip()
        output = orig_image.copy()
        cv2.rectangle(output, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.putText(output, text, (start_x, start_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        final_text.append(text)
        print(final_text)
        print(' '.join(final_text))

        cv2.imshow('Detection', output)
        cv2.waitKey(0)


if __name__ == '__main__':
    args = get_arguments()

    main(
        image=args['image'],
        width=args['width'],
        height=args['height'],
        min_confidence=args['min_confidence'],
        padding=args['padding']
    )
