import numpy as np
import tensorflow as tf

number_of_pooling_layers = 5


def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img': tf.VarLenFeature(tf.float32),
            'cell': tf.FixedLenFeature((1, 2), tf.int64),
            'bb': tf.FixedLenFeature((4,), tf.float32),
            'attr': tf.FixedLenFeature((2,), tf.float32)
        })

    image = tf.sparse_tensor_to_dense(features['img'])
    out_index = features['cell']
    bb = tf.div(features['bb'], [1.0, 1.0, 32.0, 32.0])
    gender = features['attr'][0]
    age = features['attr'][1]

    img_size = tf.cast(tf.sqrt(tf.shape(image)[0] / 3), tf.int32)
    image = tf.reshape(image, [img_size, img_size, 3])

    o_dim = tf.cast(tf.math.ceil(img_size / (2 ** number_of_pooling_layers)), tf.int64)
    prob_out = tf.expand_dims(
        tf.sparse_to_dense(sparse_indices=out_index, sparse_values=[1.0], output_shape=[o_dim, o_dim]),
        axis=-1)
    output = tf.concat([
        prob_out,
        tf.tile(prob_out, [1, 1, 4]) * bb,
        prob_out * (1.0 - gender),
        prob_out * gender,
        tf.sparse_to_dense(
            sparse_indices=tf.concat([out_index, [[tf.cast((age - 10) / 10, tf.int64)]]], axis=1),
            sparse_values=[1.0],
            output_shape=[o_dim, o_dim, 7]
        )
    ], axis=-1
    )

    return image, output


def raw_values_parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img': tf.VarLenFeature(tf.float32),
            'attr': tf.FixedLenFeature((2,), tf.float32)
        })

    image = tf.sparse_tensor_to_dense(features['img'])
    img_size = tf.cast(tf.sqrt(tf.shape(image)[0] / 3), tf.int32)
    image = tf.reshape(image, [img_size, img_size, 3])
    gender = features['attr'][0]
    age = features['attr'][1]
    return image, tf.stack([gender, age], axis=0)


def make_batch(filenames, batch_size, repeat=True, raw_values=False):
    dataset = tf.data.TFRecordDataset(filenames)
    if repeat:
        # Repeat infinitely.
        dataset = dataset.repeat()

    # Parse records.
    dataset = dataset.map(raw_values_parser if raw_values else parser, num_parallel_calls=8)
    dataset = dataset.prefetch(buffer_size=2 * batch_size)

    # dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=1)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch


def single_np_datapoint_generator(filenames, raw_values=False):
    for filename in filenames:
        for serialized_example in tf.python_io.tf_record_iterator(filename):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)

            image = np.array(example.features.feature['img'].float_list.value)
            attr = np.array(example.features.feature['attr'].float_list.value)

            img_shape = int(np.sqrt(image.shape[0] / 3))
            if raw_values:
                yield image.reshape([img_shape, img_shape, 3]), (attr[0], attr[1])
                continue

            cell = np.array(example.features.feature['cell'].int64_list.value)
            bb = np.array(example.features.feature['bb'].float_list.value)

            out_shape = int(np.ceil(img_shape / (2 ** number_of_pooling_layers)))
            output = np.zeros(shape=(out_shape, out_shape, 14))

            output[cell[0], cell[1], 0] = 1.0  # face exists
            bb[2:] = bb[2:] / 32.0
            output[cell[0], cell[1], 1:5] = bb  # dimensions and position
            output[cell[0], cell[1], 5 + int(attr[0])] = 1.0  # gender
            output[cell[0], cell[1], 7 + int((attr[1] - 10) / 10)] = 1.0  # age group

            yield (image.reshape([img_shape, img_shape, 3]), output)


def get_prediction(o, i, give_prob=True):
    age_buck = np.argmax(o[i[0], i[1], 7:])
    ret = [32 * (i[0] + o[i[0], i[1], 1]) - (o[i[0], i[1], 3] * 32.0 / 2),
           32 * (i[1] + o[i[0], i[1], 2]) - (o[i[0], i[1], 4] * 32.0 / 2),
           o[i[0], i[1], 3] * 32.0,
           o[i[0], i[1], 4] * 32.0]
    if give_prob:
        ret.append(('male (%f)' % o[i[0], i[1], 6]) if o[i[0], i[1], 6] > o[i[0], i[1], 5] else (
                'female (%f)' % o[i[0], i[1], 5]))
        ret.append('%s-%s (%f)' % (10 + age_buck * 10, 20 + age_buck * 10, o[i[0], i[1], 7 + age_buck]))
    else:
        ret.append('male' if o[i[0], i[1], 6] > o[i[0], i[1], 5] else 'female')
        ret.append('%s-%s' % (10 + age_buck * 10, 20 + age_buck * 10))
    return ret
