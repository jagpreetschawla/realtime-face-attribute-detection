import os
import tensorflow as tf
import matplotlib.ticker as ticker


def get_tf_session(check_point_path=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if check_point_path:
        saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if check_point_path:
        if os.path.exists(check_point_path + '.index'):
            saver.restore(sess, check_point_path)
            print('NOTE: restored variables from checkpoint file!')
        else:
            print('No Checkpoint exits to restore!')
        return sess, saver
    return sess


def group_iterable_into_list(iter, size, data_unroll_size=1):
    if data_unroll_size <= 1:
        l = []
        for i in iter:
            l.append(i)
            if len(l) == size:
                yield l
                l = []
        if len(l) > 0:
            yield l
    else:
        l = tuple([] for i in range(data_unroll_size))
        for i in iter:
            for j in range(len(l)):
                l[j].append(i[j])
            if len(l[0]) == size:
                yield l
                l = tuple([] for i in range(data_unroll_size))
        if len(l[0]) > 0:
            yield l


def set_axis_title_grid(ax, title, grid_size):
    ax.set_title(title)
    tick = ticker.MultipleLocator(grid_size)
    ax.xaxis.set_major_locator(tick)
    ax.yaxis.set_major_locator(tick)
    ax.grid(color='b', linestyle=':', linewidth=1)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def non_max_suppress(boxes, probs, thresh=0.2):
    b_sorted = [x for _, x in sorted(zip(probs, boxes), reverse=True)]
    filtered = []
    for b in b_sorted:
        ignore = False
        box = {'x1': b[0], 'x2': b[0] + b[2], 'y1': b[1], 'y2': b[1] + b[3]}
        for i in filtered:
            iou = get_iou(box,
                          {'x1': i[0], 'x2': i[0] + i[2], 'y1': i[1], 'y2': i[1] + i[3]})
            if iou > thresh:
                ignore = True
                break
        if not ignore:
            filtered.append(b)
    return filtered
