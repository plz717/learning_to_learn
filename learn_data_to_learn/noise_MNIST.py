from tensorflow.examples.tutorials.mnist import input_data
import random
import pickle


def split_into_10_folds(dataset):
    part_num = int(dataset.images.shape[0] / 10)
    splited_images = []
    splited_labels = []
    for i in range(10):
        images, labels = dataset.next_batch(part_num, shuffle=True)
        splited_images.append(images)
        splited_labels.append(labels)
    return splited_images, splited_labels


def random_flip(splited_set):
    folds_num = len(splited_set)
    for i in range(1, folds_num + 1):
        for img in splited_set[i - 1]:
            img_size = img.shape[0]
            flip_num = int((i - 1) * 0.1 * img_size)
            print("flip_num is:{}".format(flip_num))
            flip_idx = random.sample(range(img_size), flip_num)
            # img_flipped = [1.0 - item for index, item in enumerate(img) if
            # index in flip_idx else item]
            for index, item in enumerate(img):
                if index in flip_idx:
                    print("i will flip")
                    img[index] = 1.0 - item


if __name__ == '__main__':
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    train_x = mnist.train.images
    train_y = mnist.train.labels
    test_x = mnist.test.images
    test_y = mnist.train.labels
    dataset = mnist.train
    splited_train_x, splited_train_y = split_into_10_folds(dataset)
    random_flip(splited_train_x)

    with open('../flipped_train_x', 'wb') as f:
        pickle.dump(splited_train_x, f)
    with open('../flipped_train_y', 'wb') as f:
        pickle.dump(splited_train_y, f)
