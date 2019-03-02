import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from data_utils import get_prediction
from utils import non_max_suppress

REQUIRED_INPUT_SIZE = 250  # depends on model
DISPLAY_SIZE = 600

def resize_n_pad(img, to_size=250):
    if img.shape[0] > img.shape[1]:
        sy = to_size
        sx = int((img.shape[1] / img.shape[0]) * to_size)
    else:
        sx = to_size
        sy = int((img.shape[0] / img.shape[1]) * to_size)
    img = cv2.resize(img, dsize=(sx, sy), interpolation=cv2.INTER_AREA)
    img = np.pad(img, ((0, to_size - img.shape[0]), (0, to_size - img.shape[1]), (0, 0)), "constant")
    return img


def display(img, pred, scale_fac=1.0):
    found_map = pred[:, :, 0] > 0.7
    found = np.argwhere(found_map)
    blist = []
    probs = []
    for i in found:
        y, x, h, w, g, a = get_prediction(pred, i)
        blist.append((x, y, w, h, g, a))
        probs.append(pred[i[0], i[1], 0])
    filtered = non_max_suppress(blist, probs)
    for x, y, w, h, g, a in filtered:
        cv2.rectangle(img,
                      (int(scale_fac * x), int(scale_fac * y)),
                      (int(scale_fac * (x + w)), int(scale_fac * (y + h))),
                      color=(255, 0, 0), thickness=2)
        cv2.putText(img, g, (int(scale_fac * x), int(scale_fac * y - 10)),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
        cv2.putText(img, a, (int(scale_fac * x), int(scale_fac * (y + h + 10))),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0, 255), thickness=2)
    cv2.imshow("probabilities", cv2.resize(pred[:, :, 0], (240, 240), interpolation=cv2.INTER_NEAREST))
    cv2.imshow("Live detect", img)
    cv2.waitKey(1)


def main():
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        print("Loading model...")
        tf.saved_model.loader.load(sess, [tag_constants.SERVING], "./models/test/yolo_saved_model")
        inp = graph.get_tensor_by_name('Input:0')
        out = graph.get_tensor_by_name('Output:0')
        print("Model loaded")

        print("opening camera...")
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            frame = cv2.normalize(frame.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            frame = resize_n_pad(frame, DISPLAY_SIZE)
            inp_img = cv2.resize(frame, dsize=(REQUIRED_INPUT_SIZE, REQUIRED_INPUT_SIZE), interpolation=cv2.INTER_AREA)
            pred = sess.run(out, feed_dict={inp: [inp_img]})
            display(frame, pred[0], DISPLAY_SIZE/REQUIRED_INPUT_SIZE)


if __name__ == "__main__":
    main()
