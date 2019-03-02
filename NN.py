import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import patches

from data_utils import single_np_datapoint_generator, get_prediction, number_of_pooling_layers
from utils import group_iterable_into_list, set_axis_title_grid, get_iou, non_max_suppress


def _base_NN(X, no_of_filters_fns, no_of_pooling=number_of_pooling_layers):
    layer = X
    layers = []
    for i in range(no_of_pooling):
        for fn_ind in range(len(no_of_filters_fns)):
            num = no_of_filters_fns[fn_ind](i)
            if num is None or num == 0:
                continue
            layer = tf.layers.conv2d(layer, filters=num, kernel_size=(3, 3), padding='SAME',
                                     activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name="conv%d_%d" % (i, fn_ind))
            layers.append(layer)
        layer = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=2, padding='SAME', name="pool%d" % i)
        layers.append(layer)
    return layers


def get_regression_NN(X, image_size, no_of_conv_filters_fns, no_of_pooling, no_of_outputs):
    layers = _base_NN(X, no_of_conv_filters_fns, no_of_pooling)
    last_layer_flat = tf.reshape(layers[-1], (-1, (int(np.ceil(image_size/(2**no_of_pooling)))**2)*layers[-1].shape[-1]))
    dense = tf.layers.dense(inputs=last_layer_flat, units=no_of_outputs, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
    return dense


def get_classification_NN(X, image_size, no_of_conv_filters_fns, no_of_pooling, no_of_outputs):
    logits = get_regression_NN(X, image_size, no_of_conv_filters_fns, no_of_pooling, no_of_outputs)
    return tf.nn.softmax(logits), logits


def get_yolo_NN(X, no_of_filter_fns):
    layers = _base_NN(X, no_of_filter_fns)
    layer = layers[-1]
    dim = tf.clip_by_value(
        tf.layers.conv2d(layer, filters=2, kernel_size=(1, 1), padding='SAME', activation=tf.exp, reuse=tf.AUTO_REUSE,
                         name="conv_dim"), 0, 1000)
    box = tf.layers.conv2d(layer, filters=2, kernel_size=(1, 1), padding='SAME', activation=tf.nn.sigmoid,
                           reuse=tf.AUTO_REUSE, name="conv_box")
    face = tf.layers.conv2d(layer, filters=1, kernel_size=(1, 1), padding='SAME', activation=tf.nn.sigmoid,
                            reuse=tf.AUTO_REUSE, name="conv_face")

    gender_logits = tf.layers.conv2d(layer, filters=2, kernel_size=(1, 1), padding='SAME', activation=tf.nn.relu,
                                     reuse=tf.AUTO_REUSE, name="conv_gender")
    age_logits = tf.layers.conv2d(layer, filters=7, kernel_size=(1, 1), padding='SAME', activation=tf.nn.relu,
                                  reuse=tf.AUTO_REUSE, name="conv_age")

    ##pred = tf.concat([dim, box, tf.nn.softmax(gender_logits), tf.nn.softmax(age_logits)], axis=3)
    pred = tf.concat([face, box, dim, tf.nn.softmax(gender_logits), tf.nn.softmax(age_logits)], axis=3)
    return layers, [gender_logits, age_logits], pred


def get_training_op(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0008)
    train_op = optimizer.minimize(loss=loss)
    return train_op


def create_yolo_model_graph(X, Y, no_of_filter_fns, loss_weights, gender_weights=(1.0,) * 2, age_weights=(1.0,) * 7):
    _, logits, pred = get_yolo_NN(X, no_of_filter_fns)
    loss = loss_weights[0] * tf.losses.mean_squared_error(Y[:, :, :, 3], pred[:, :, :, 3], weights=Y[:, :, :, 0]) \
           + loss_weights[0] * tf.losses.mean_squared_error(Y[:, :, :, 4], pred[:, :, :, 4], weights=Y[:, :, :, 0]) \
           + loss_weights[1] * tf.losses.mean_squared_error(Y[:, :, :, 1], pred[:, :, :, 1], weights=Y[:, :, :, 0]) \
           + loss_weights[1] * tf.losses.mean_squared_error(Y[:, :, :, 2], pred[:, :, :, 2], weights=Y[:, :, :, 0]) \
           + loss_weights[2] * tf.losses.log_loss(Y[:, :, :, 0], pred[:, :, :, 0], weights=Y[:, :, :, 0]) \
           + loss_weights[3] * tf.losses.log_loss(Y[:, :, :, 0], pred[:, :, :, 0], weights=1.0 - Y[:, :, :, 0]) \
           + loss_weights[4] * tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(Y[:, :, :, 5:7], axis=3),
                                                                      logits=logits[0],
                                                                      weights=tf.add_n([
                                                                          gender_weights[0] * Y[:, :, :, 5],
                                                                          gender_weights[1] * Y[:, :, :, 6]
                                                                      ])) \
           + loss_weights[5] * tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(Y[:, :, :, 7:], axis=3),
                                                                      logits=logits[1],
                                                                      weights=tf.add_n([
                                                                          age_weights[0]*Y[:, :, :, 7],
                                                                          age_weights[1]*Y[:, :, :, 8],
                                                                          age_weights[2]*Y[:, :, :, 9],
                                                                          age_weights[3]*Y[:, :, :, 10],
                                                                          age_weights[4]*Y[:, :, :, 11],
                                                                          age_weights[5]*Y[:, :, :, 12],
                                                                          age_weights[6]*Y[:, :, :, 13]
                                                                      ]))
    return get_training_op(loss), loss


def draw_layer_output(sess, X, layer, inputs, show_colorbar=False):
    outp = sess.run(layer, feed_dict={X: inputs})
    per_inp = outp.shape[3]
    total_inp = outp.shape[0]
    fig, ax = plt.subplots(ncols=per_inp, nrows=total_inp)
    if 1 in [per_inp, total_inp]:
        ax = [ax]
    fig.set_figwidth(25), fig.set_figheight(total_inp * 5)
    for i in range(total_inp):
        for j in range(per_inp):
            im = ax[i][j].imshow(outp[i, :, :, j], cmap="gray")
            if show_colorbar:
                plt.colorbar(im, ax=ax[i][j])


def see_and_compare_yolo_outputs(sess, X, pred, img, out):
    o = sess.run([pred], feed_dict={X: [img]})[0][0]

    fig, ax = plt.subplots(ncols=2, nrows=2)
    fig.set_figheight(12), fig.set_figwidth(12)

    ax[0][0].imshow(img, extent=(0, 250, 250, 0), interpolation='none')
    set_axis_title_grid(ax[0][0], 'Actual Output', 32)
    found = np.argwhere(out[:, :, 0] > 0.6)
    for i in found:
        y, x, h, w, g, a = get_prediction(out, i)
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax[0][0].text(x, y - 25, g, backgroundcolor='r')
        ax[0][0].text(x, y - 10, a, backgroundcolor='r')
        ax[0][0].add_patch(rect)

    ax[0][1].imshow(img, extent=(0, 250, 250, 0), interpolation='none')
    set_axis_title_grid(ax[0][1], 'NN Output', 32)
    found_map = o[:, :, 0] > 0.7
    found = np.argwhere(found_map)

    blist = []
    probs = []
    for i in found:
        y, x, h, w, g, a = get_prediction(o, i)
        blist.append((x, y, w, h, g, a))
        probs.append(o[i[0], i[1], 0])
    filtered = non_max_suppress(blist, probs)
    for x, y, w, h, g, a in filtered:
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax[0][1].text(x, y - 25, g, backgroundcolor='r')
        ax[0][1].text(x, y - 10, a, backgroundcolor='r')
        ax[0][1].add_patch(rect)

    ax[1][0].imshow(o[:, :, 0], cmap='gray', extent=(0, 7, 7, 0), interpolation='none')
    set_axis_title_grid(ax[1][0], 'Predicted Probabilities', 1)
    ax[1][1].imshow(found_map, cmap='gray', extent=(0, 7, 7, 0), interpolation='none')
    set_axis_title_grid(ax[1][1], 'Predicted Probabilities Above Threshold', 1)
    plt.show()


def calculate_yolo_metric(sess, X, pred, data_files, batch_size=100):
    count = 0
    iou_total = 0
    wrong_area = 0
    correct_g = 0
    correct_a = 0
    img_iter = single_np_datapoint_generator(data_files)
    for imgs, outs in group_iterable_into_list(img_iter, batch_size, 2):
        preds = sess.run(pred, feed_dict={X: imgs})
        for c in range(len(imgs)):
            out = outs[c]
            o = preds[c]
            orig = np.argwhere(out[:, :, 0] > 0.6)
            y_o, x_o, h_o, w_o, g_o, a_o = get_prediction(out, orig[0], give_prob=False)  # assuming only one face
            found = np.argwhere(o[:, :, 0] > 0.7)
            blist = []
            probs = []
            for i in found:
                y, x, h, w, g, a = get_prediction(o, i, give_prob=False)
                blist.append((x, y, w, h, g, a))
                probs.append(o[i[0], i[1], 0])
            filtered = non_max_suppress(blist, probs)
            max_iou = -1
            o_box = {'x1': x_o, 'x2': x_o + w_o, 'y1': y_o, 'y2': y_o + h_o}
            g_p, a_p = '', ''
            for x, y, w, h, g, a in filtered:
                iou = get_iou(o_box,
                              {'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h})
                if iou > 0.2 and iou > max_iou:
                    if max_iou > 0:
                        wrong_area += 1.0 - max_iou
                    max_iou = iou
                    g_p = g
                    a_p = a
                else:
                    wrong_area += 1.0 - iou
            if max_iou > 0:
                iou_total += max_iou
                if g_o == g_p:
                    correct_g += 1.0
                if a_o == a_p:
                    correct_a += 1.0
            count += 1
        print("Processed", count, "images")
    return {
        "total_count": count,
        "average_iou": iou_total / count,
        "average_false_iou": wrong_area / count,
        "gender_accuracy": correct_g / count,
        "age_accuracy": correct_a / count
    }
