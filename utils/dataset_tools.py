import os
import shutil
from sklearn import metrics
import numpy as np


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(np.int)

    try:
        acc = metrics.accuracy_score(target, pred)
    except:
        acc = 0
        print("Input contains NaN, infinity or a value too large for dtype('float32').")

    try:
        auroc = metrics.roc_auc_score(target, probas_pred)
    except:
        auroc = 0
        print("Input contains NaN, infinity or a value too large for dtype('float32').")

    try:
        f1_score = metrics.f1_score(target, pred)
    except:
        f1_score = 0
        print("Input contains NaN, infinity or a value too large for dtype('float32').")


    try:
        precision = metrics.precision_score(target, pred)
    except:
        precision = 0
        print("Input contains NaN, infinity or a value too large for dtype('float32').")


    try:
        recall = metrics.recall_score(target, pred)
    except:
        recall = 0
        print("Input contains NaN, infinity or a value too large for dtype('float32').")

    try:
        ap= metrics.average_precision_score(target, probas_pred)
    except:
        ap = 0
        print("Input contains NaN, infinity or a value too large for dtype('float32').")

    return acc, auroc, f1_score, precision, recall, ap

def maybe_unzip_dataset(args):

    datasets = [args.dataset_name]
    dataset_paths = [args.dataset_path]
    done = False

    for dataset_idx, dataset_path in enumerate(dataset_paths):
        if dataset_path.endswith('/'):
            dataset_path = dataset_path[:-1]
        print(dataset_path)
        if not os.path.exists(dataset_path):
            print("Not found dataset folder structure.. searching for .tar.bz2 file")
            zip_directory = "{}.tar.bz2".format(os.path.join("~/meta_ddi/MeTAL-master/datasets/", datasets[dataset_idx]))

            assert os.path.exists(os.path.abspath(zip_directory)), "{} dataset zip file not found" \
                                                  "place dataset in datasets folder as explained in README".format(os.path.abspath(zip_directory))
            print("Found zip file, unpacking")

            unzip_file(filepath_pack=os.path.join("~/meta_ddi/MeTAL-master/datasets/", "{}.tar.bz2".format(datasets[dataset_idx])),
                       filepath_to_store="~/meta_ddi/MeTAL-master/datasets/")



            args.reset_stored_filepaths = True

        total_files = 0
        for subdir, dir, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") or file.lower().endswith(
                        ".png") or file.lower().endswith(".pkl"):
                    total_files += 1
        print("count stuff________________________________________", total_files)
        if (total_files == 1623 * 20 and datasets[dataset_idx] == 'omniglot_dataset') or (
                total_files == 100 * 600 and 'mini_imagenet' in datasets[dataset_idx]) or (
                total_files == 3 and 'mini_imagenet_pkl' in datasets[dataset_idx]):
            print("file count is correct")
            done = True
        elif datasets[dataset_idx] not in [
            'omniglot_dataset',
            'mini_imagenet',
            'mini_imagenet_pkl',
        ]:
            done = True
            print("using new dataset")

        if not done:
            shutil.rmtree(dataset_path, ignore_errors=True)
            maybe_unzip_dataset(args)


def unzip_file(filepath_pack, filepath_to_store):
    command_to_run = "tar -I pbzip2 -xf {} -C {}".format(filepath_pack, filepath_to_store)
    os.system(command_to_run)
