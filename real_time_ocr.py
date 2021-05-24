########################################################################################################################
# Filename: dynamic_text_recognition.py
# Description: Recognizes text from video or webcam
# Usage: python dynamic_text_recognition.py
#        (python dynamic_text_recognition.py -v | -c | -w | -e | -p (arguments))
########################################################################################################################

import cv2
import time
import argparse
import pytesseract
import numpy as np

import imutils
from imutils.video import VideoStream, FPS
from imutils.object_detection import non_max_suppression

from utils import box_extractor

# setting up tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# setting EAST module
east = 'frozen_east_text_detection.pb'


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', type=str,
                    help='path to optional video file')
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


if __name__ == '__main__':

    args = get_arguments()

    w, h = None, None
    new_w, new_h = args['width'], args['height']
    ratio_w, ratio_h = None, None

    layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east)

    if not args.get('video', False):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(1)

    else:
        vs = cv2.VideoCapture(args['video'])

    fps = FPS().start()

    while True:

        frame = vs.read()
        frame = frame[1] if args.get('video', False) else frame

        if frame is None:
            break

        frame = imutils.resize(frame, width=1000)
        orig = frame.copy()
        orig_h, orig_w = orig.shape[:2]

        if w is None or h is None:
            h, w = frame.shape[:2]
            ratio_w = w / float(new_w)
            ratio_h = h / float(new_h)

        frame = cv2.resize(frame, (new_w, new_h))

        blob = cv2.dnn.blobFromImage(frame, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        scores, geometry = net.forward(layer_names)

        rectangles, confidences = box_extractor(
            scores, geometry, min_confidence=args['min_confidence'])
        boxes = non_max_suppression(np.array(rectangles), probs=confidences)

        for (start_x, start_y, end_x, end_y) in boxes:

            start_x = int(start_x * ratio_w)
            start_y = int(start_y * ratio_h)
            end_x = int(end_x * ratio_w)
            end_y = int(end_y * ratio_h)

            dx = int((end_x - start_x) * args['padding'])
            dy = int((end_y - start_y) * args['padding'])

            start_x = max(0, start_x - dx)
            start_y = max(0, start_y - dy)
            end_x = min(orig_w, end_x + (dx * 2))
            end_y = min(orig_h, end_y + (dy * 2))

            # ROI to be recognized
            roi = orig[start_y:end_y, start_x:end_x]

            # recognizing text
            config = '-l eng --oem 1 --psm 7'
            text = pytesseract.image_to_string(roi, config=config)

            cv2.rectangle(orig, (start_x, start_y),
                          (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(orig, text, (start_x, start_y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        fps.update()

        cv2.imshow("Detection", orig)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    fps.stop()
    print(f"[INFO] elapsed time {round(fps.elapsed(), 2)}")
    print(f"[INFO] approx. FPS : {round(fps.fps(), 2)}")

    if not args.get('video', False):
        vs.stop()

    else:
        vs.release()

    cv2.destroyAllWindows()
