import pickle
import cv2
import os
import pandas as pd
import numpy as np

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def unpickle(file_path, img_dir):
    print(f"Read data from file: {file_path}")
    with open(file_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

    filenames = data_dict[b'filenames']
    data = data_dict[b'data']
    labels = data_dict[b'labels']
    samples = []

    for file, arr, label in zip(filenames, data, labels):
        file = file.decode("utf-8")

        file_path = f"{img_dir}/{file}"
        img = arr.reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))

        cv2.imwrite(file_path, img)

        label_name = label_names[label]
        samples.append({"file_path": file_path, "label_name": label_name, "label": label})

    return samples


def read_train(img_dir, out_dir):
    full_batchs = [
        "data/cifar-10-batches-py/data_batch_1",
        "data/cifar-10-batches-py/data_batch_2",
        "data/cifar-10-batches-py/data_batch_3",
        "data/cifar-10-batches-py/data_batch_4",
        "data/cifar-10-batches-py/data_batch_5"
    ]

    full_samples = []
    for batch_dir in full_batchs:
        samples = unpickle(batch_dir, img_dir)
        full_samples.extend(samples)

    df = pd.DataFrame(full_samples)
    df.to_csv(f"{out_dir}/train.csv", index=False)

    print("Total train sample", len(df))

def read_test(img_dir, out_dir):
    full_batchs = [
        "data/cifar-10-batches-py/test_batch"
    ]

    full_samples = []
    for batch_dir in full_batchs:
        samples = unpickle(batch_dir, img_dir)
        full_samples.extend(samples)

    df = pd.DataFrame(full_samples)
    df.to_csv(f"{out_dir}/test.csv", index=False)
    print("Total test sample", len(df))

if __name__ == '__main__':
    img_dir = "data/images"
    out_dir = "data/dataset"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    read_train(img_dir, out_dir)
    read_test(img_dir, out_dir)