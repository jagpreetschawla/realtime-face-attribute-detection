import argparse
import glob
import math
import tarfile
import tempfile
from datetime import datetime, timedelta
from os import remove
from os import sep as os_path_sep
from os.path import isdir, join, exists, splitext, relpath

parser = argparse.ArgumentParser(
    description="This program processes imdb-wiki face dataset to produce a tfrecord file for training a YOLO style NN. " +
                "The tfrecord files will have fixed size input images and yolo style output grid containing bounding boxes, gender and age")
parser.add_argument("mat_file", help="file containing metadata for imdb-wiki face data.")
parser.add_argument("data_paths", nargs='+', help="directory or tar file containing imdb-wiki face data.")
parser.add_argument("-o", "--output_path", help="directory to write output tfrecord files", required=True)
parser.add_argument("-t", "--temp_path", help="Path to base directory to use for creating temp files", default=None)
parser.add_argument("-c", "--grid_cell_size",
                    help="define how many pixels in input should be 1 grid cell in yolo output.",
                    type=int, default=32)
parser.add_argument("-s", "--input_size",
                    help="defines the number of pixels in each row or column you need in input image (image will have equal width and height).",
                    type=int, default=250)
parser.add_argument("-m", "--minimum_size",
                    help="minimum image width/height. Images smaller than this will be ignored.",
                    type=int, default=150)
parser.add_argument("--create_cropped",
                    help="flag to also create a dataset with cropped face images",
                    action="store_true")
parser.add_argument("--no_full_size",
                    help="flag to not create full image dataset. This only makes sense when you have flag set to create cropped faces.",
                    action="store_false")
parser.add_argument("-f", "--face_size",
                    help="defines the number of pixels in each row or column you need in Cropped Face input image (image will have equal width and height)",
                    type=int, default=50)
args = parser.parse_args()

mat_file = args.mat_file
data_paths = args.data_paths
REDUCE_BY = args.grid_cell_size
REQ_SIZE = args.input_size
MIN_SIZE = args.minimum_size
OUT_FILE_PATH = args.output_path
TEMP_DIR = args.temp_path
CREATE_CROPPED = args.create_cropped
CREATE_FULL = args.no_full_size
FACE_SIZE = args.face_size

for i in data_paths:
    if not exists(i) or (not (i.endswith(".tar.gz") or i.endswith(".tar")) and not isdir(i)):
        parser.error("Please provide a valid directory or tar file")

# importing it later when required as it takes time
print("Importing libraries and frameworks..")
import numpy as np
import tensorflow as tf
import scipy.io as sio
# from scipy.ndimage import imread
# from matplotlib.pyplot import imread
from skimage.io import imread
# from scipy.misc import imresize
from skimage.transform import resize as imresize
from skimage.color import gray2rgb
from skimage import img_as_float64
from multiprocessing import Process, Queue, cpu_count

temp_d = tempfile.TemporaryDirectory(prefix="dataset_to_tf_", suffix=".temp", dir=TEMP_DIR)


def files(tar_or_dir, ext, shuffle=False):
    if isdir(tar_or_dir):
        pat = join(tar_or_dir, "**", "*." + ext)
        it = np.random.permutation(glob.glob(pat, recursive=True)) if shuffle else glob.iglob(pat, recursive=True)
        for f in it:
            path = relpath(f, start=tar_or_dir)
            if path.startswith("imdb") or path.startswith("wiki"):
                path = path[5:]
            if os_path_sep != "/":
                path = path.replace(os_path_sep, "/")
            yield path, f
    else:
        tar = tarfile.open(tar_or_dir)
        it = np.random.permutation(tar.getmembers()) if shuffle else tar
        for tarinfo in it:
            if splitext(tarinfo.name)[1] == "." + ext:
                name = tarinfo.name
                if name.startswith("imdb") or name.startswith("wiki"):
                    name = name[5:]
                tar.extract(tarinfo, path=temp_d.name)
                yield name, join(temp_d.name, tarinfo.name)


def mat_to_py_time(mat_time):
    mat_time = float(mat_time)
    return datetime.fromordinal(int(mat_time)) + timedelta(days=mat_time % 1) - timedelta(days=366)


def process_and_save(shared_q, index, res_q):
    if CREATE_FULL:
        writer = tf.python_io.TFRecordWriter(join(OUT_FILE_PATH, "%d.tfrecord" % index))
    if CREATE_CROPPED:
        writer_cropped = tf.python_io.TFRecordWriter(join(OUT_FILE_PATH, "%d_cropped.tfrecord" % index))
    count = 0
    while True:
        i = shared_q.get()
        if i is None:
            if CREATE_FULL:
                writer.close()
            if CREATE_CROPPED:
                writer_cropped.close()
            res_q.put(count)
            return
        f, gender, age, face_loc = i
        try:
            x = imread(f, img_num=0)
        except:
            print("Unable to read", f)
            continue
        if len(x.shape) < 3:
            x = gray2rgb(x)
        y = np.floor(face_loc) - 1.0

        if x.shape[0] < MIN_SIZE or x.shape[1] < MIN_SIZE:
            continue
        if math.isnan(gender) or math.isnan(age):
            continue
        if age < 10 or age >= 80:
            continue
        if abs(y[3] - y[1]) > x.shape[0] or abs(y[2] - y[0]) > x.shape[1]:
            continue
        if any(p < 0 for p in [y[0], y[1], y[2], y[3]]) or \
                any(p > x.shape[1] for p in [y[0], y[2]]) or \
                any(p > x.shape[0] for p in [y[1], y[3]]):
            continue

        if CREATE_FULL:
            if x.shape[0] > x.shape[1]:
                sy = REQ_SIZE
                sx = int((float(x.shape[1]) / x.shape[0]) * REQ_SIZE)
            else:
                sx = REQ_SIZE
                sy = int((float(x.shape[0]) / x.shape[1]) * REQ_SIZE)
            out = np.zeros(y.shape)
            out[0] = (float(y[0]) / x.shape[1]) * sx
            out[1] = (float(y[1]) / x.shape[0]) * sy
            out[2] = (float(y[2]) / x.shape[1]) * sx
            out[3] = (float(y[3]) / x.shape[0]) * sy
            out = np.floor(out)
            img = imresize(x, (sy, sx), mode="constant", anti_aliasing=True)

            img = img_as_float64(img)
            img = np.pad(img, ((0, REQ_SIZE - img.shape[0]), (0, REQ_SIZE - img.shape[1]), (0, 0)), "constant")

            block_x = int(math.ceil(img.shape[1] / REDUCE_BY))
            block_y = int(math.ceil(img.shape[0] / REDUCE_BY))
            center = ((out[0] + out[2]) / 2.0, (out[1] + out[3]) / 2.0)
            block_index = (int(math.floor(center[0] / REDUCE_BY)), int(math.floor(center[1] / REDUCE_BY)))

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'img': tf.train.Feature(
                            float_list=tf.train.FloatList(value=img.reshape(-1))),
                        'cell': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[block_index[1], block_index[0]])),
                        # cell containing face
                        'bb': tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=[center[1] / REDUCE_BY - block_index[1],  # center within grid cell
                                       center[0] / REDUCE_BY - block_index[0],
                                       out[3] - out[1],  # height
                                       out[2] - out[0]])),  # width
                        'attr': tf.train.Feature(  # face attributes
                            float_list=tf.train.FloatList(value=[float(gender), float(age)]))
                    })
            )
            serialized = example.SerializeToString()
            writer.write(serialized)

        if CREATE_CROPPED:
            img = imresize(x[int(y[1]):int(y[3]), int(y[0]):int(y[2])], (FACE_SIZE, FACE_SIZE), mode="constant",
                           anti_aliasing=True)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'img': tf.train.Feature(
                            float_list=tf.train.FloatList(value=img.reshape(-1).astype("float"))),
                        'attr': tf.train.Feature(
                            float_list=tf.train.FloatList(value=[float(gender), float(age)]))
                    })
            )
            serialized = example.SerializeToString()
            writer_cropped.write(serialized)

        count += 1
        if f.startswith(temp_d.name):  # in temp folder, was extracted from tar file
            remove(f)


def producer(dirs, shared_q, count_q=None):
    count = 0
    for tar_or_dir in dirs:
        for path, f in files(tar_or_dir, 'jpg'):
            loc = np.where(m.full_path == path)
            if len(loc) != 1 or len(loc[0]) == 0:
                print(path, "Not found in metadata")
                continue
            i = loc[0][0]
            try:
                if math.isnan(m.face_score[i]) or math.isinf(m.face_score[i]) or not math.isnan(
                        m.second_face_score[i]) or m.face_score[i] < 1.5:
                    continue
                print("Queuing file", path, "for processing")
                shared_q.put((f, m.gender[i], m.photo_taken[i] - mat_to_py_time(m.dob[i]).year, m.face_location[i]))
                count += 1
            except Exception as e:
                print("Something went wrong while reading data for", path, ": ", str(e))
    if count_q:
        count_q.put(count)
    else:
        return count


if __name__ == "__main__":
    # start processes for parallel processing
    total_threads = 10
    shared_q = Queue(total_threads)
    res_q = Queue(2)
    processes = []
    print("Starting parallel processes..")
    for i in range(total_threads):
        p = Process(target=process_and_save, args=(shared_q, i, res_q))
        p.start()
        processes.append(p)

    # this needs to be done after making processes as we don't want processes to have a copy of this large file.
    m = sio.loadmat(mat_file, struct_as_record=False, squeeze_me=True)

    if 'wiki' in m:
        m = m['wiki']
    elif 'imdb' in m:
        m = m['imdb']
    else:
        # wait for all processes to exit
        for i in range(len(processes)):
            shared_q.put(None)
        for p in processes:
            p.join()
        parser.error("Please provide a valid mat file!")

    np.random.seed()
    dirs = np.random.permutation(data_paths)
    print("Starting producers...")
    if len(dirs) > 1:
        count_q = Queue(2)
        producers = []
        no_of_producers = min(2, len(dirs))
        per_prod = len(dirs) // no_of_producers
        for i in range(no_of_producers):
            p = Process(target=producer, args=(dirs[i * per_prod: i * per_prod + per_prod], shared_q, count_q))
            p.start()
            producers.append(p)

        for p in producers:
            p.join()

        count = 0
        for i in range(len(processes)):
            count += count_q.get()
    else:
        count = producer(dirs, shared_q)

    # wait for all processes to exit
    print("Waiting for processes to terminate...")
    for i in range(len(processes)):
        shared_q.put(None)
    tcount = 0
    for i in range(len(processes)):
        tcount += res_q.get()
    for p in processes:
        p.join()

    print("total files which had only 1 face:", count)
    print("total_written", tcount)

    temp_d.cleanup()
