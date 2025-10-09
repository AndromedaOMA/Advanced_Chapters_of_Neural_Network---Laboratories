from torch.utils.data import Dataset

import pickle
import os
import pandas as pd
import numpy as np
import time
np.seterr(divide='ignore', invalid='ignore')


class ExtendedMNISTDataset(Dataset):
    def __init__(self, root: str = "/kaggle/input/fii-atnn-2025-competition-1", train: bool = True):
        file = "extended_mnist_test.pkl"
        if train:
            file = "extended_mnist_train.pkl"
        local_root = './data'
        # file = os.path.join(root, file)
        file = os.path.join(local_root, file)
        with open(file, "rb") as fp:
            self.data = pickle.load(fp)

    def __len__(self, ) -> int:
        return len(self.data)

    def __getitem__(self, i: int):
        image, label = self.data[i]
        # Flatten the image and normalize like download_mnist
        image = np.array(image).flatten()
        image = (image / 255.0 * 0.99) + 0.01
        return image, label


def load_extended_mnist(train=True):
    dataset = ExtendedMNISTDataset(train=train)
    images = []
    labels = []

    for image, label in dataset:
        images.append(image)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # One-hot encode labels
    num_classes = 10
    one_hot_labels = np.zeros((len(labels), num_classes))
    one_hot_labels[np.arange(len(labels)), labels] = 1

    return images, one_hot_labels


train_data, train_labels = load_extended_mnist(train=True)
test_data, test_labels = load_extended_mnist(train=False)


def batches_generator(train_data, train_labels, batch_size):
    """YIELD (continuous "return") the current batches of 100 elements each"""
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    for i in range(0, len(train_data), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield train_data[batch_indices], train_labels[batch_indices]


def sigmoid(x, backpropagation=False):
    # Clip values to prevent overflow
    x = np.clip(x, -500, 500)  # Clip x to avoid large values

    s = 1 / (1 + np.exp(-x))
    if backpropagation:
        return s * (1 - s)
    return s


def relu(x, backpropagation=False):
    if backpropagation:
        return (x > 0).astype(float)
    return np.maximum(0, x)


def softmax(x, backpropagation=False):
    # # Clip values to prevent overflow
    # x = np.clip(x, -500, 500)  # Clip x to avoid large values

    exp = np.exp(x - np.max(x))
    s = exp / np.sum(exp, axis=0, keepdims=True)
    if backpropagation:
        return s * (1 - s)
    return s


# The MLP architecture should consist of 784 input neurons, 100 hidden neurons, and 10 output neurons.
class MLP:
    def __init__(self, in_layer=784, h_layer=100, out_layer=10, lr=0.1, epochs=400, batches=250, dropout_rate=0.1):
        self.in_layer = in_layer
        self.h_layer = h_layer
        self.out_layer = out_layer
        self.lr = lr
        self.epochs = epochs
        self.batches = batches
        self.dropout_rate = dropout_rate
        '''Xavier initialization - for Logistic Activation Functions'''
        self.params_xavier = {
            'W1': np.random.randn(self.h_layer, self.in_layer) * np.sqrt(2 / (self.h_layer + self.in_layer)),
            'B1': np.zeros((h_layer, 1)),
            'W2': np.random.randn(self.out_layer, self.h_layer) * np.sqrt(2 / (self.h_layer + self.out_layer)),
            'B2': np.zeros((out_layer, 1))
        }
        '''He initialization - for Variants of ReLU Activation Functions'''
        self.params_he = {
            'W1': np.random.randn(self.h_layer, self.in_layer) * np.sqrt(2 / self.in_layer),
            'B1': np.zeros((h_layer, 1)),
            'W2': np.random.randn(self.out_layer, self.h_layer) * np.sqrt(2 / self.h_layer),
            'B2': np.zeros((out_layer, 1))
        }

    def forward_prop(self, train_data, train=True):
        params = self.params_he

        params['A0'] = train_data
        params['Z1'] = np.add(np.dot(params['W1'], params['A0'].T), params['B1'])
        # params['Z1'] = np.dot(params['W1'], params['A0'].T) + params['B1']
        params['A1'] = relu(params['Z1'])

        if train:  ## Dropout
            # Step 1: initialize matrix dropout_mask = np.random.rand(..., ...)
            dropout_mask = np.random.rand(*params['A1'].shape)
            # Step 2: convert entries of dropout_mask to 0 or 1 (using dropout_mask as the threshold)
            dropout_mask = dropout_mask < (1 - self.dropout_rate)
            # Step 3: shut down some neurons of A1
            params['A1'] = np.multiply(params['A1'], dropout_mask)
            # Step 4: scale the value of neurons that haven't been shut down
            params['A1'] /= (1 - self.dropout_rate)
            params['dropout_mask'] = dropout_mask

        params['Z2'] = np.add(np.dot(params['W2'], params['A1']), params['B2'])
        # params['Z2'] = np.dot(params['W2'], params['A1']) + params['B2']
        params['A2'] = softmax(params['Z2'])

        return params['A2']

    def backward_prop(self, prediction, train_labels):
        params = self.params_he

        # err = (prediction - train_labels.T) * softmax(params['Z2'], backpropagation=True)
        err = (prediction - train_labels.T)
        params['W2'] -= self.lr * np.dot(err, params['A1'].T) / len(train_labels)
        params['B2'] -= self.lr * np.mean(err, axis=1, keepdims=True)

        err = np.dot(params['W2'].T, err) * relu(params['Z1'], backpropagation=True)
        # if 'dropout_mask' in params:
        #     err = np.dot(err, params['dropout_mask'].T) / (1 - self.dropout_rate)
        params['W1'] -= self.lr * np.dot(err, params['A0']) / len(train_labels)
        params['B1'] -= self.lr * np.mean(err, axis=1, keepdims=True)

    def compute_acc(self, data, labels):
        predictions = self.forward_prop(np.array(data), train=False)
        prediction_labels = np.argmax(predictions, axis=0)
        true_labels = np.argmax(labels, axis=1)
        return np.mean(prediction_labels == true_labels)

    def train(self, train_data, train_labels, test_data, test_labels):
        start_time = time.time()
        for epoch in range(self.epochs):
            for data_batch, label_batch in batches_generator(train_data, train_labels, self.batches):
                prediction = self.forward_prop(data_batch)
                self.backward_prop(prediction, label_batch)
            test_accuracy = self.compute_acc(test_data, test_labels)
            print(f'Epoch: {epoch + 1}, Time Spent: {time.time() - start_time}s, Test Accuracy: {test_accuracy * 100}%')


if __name__ == '__main__':
    # print(train_data[0], train_labels[0], test_data[0], test_labels[0])
    mlp = MLP()
    mlp.train(train_data, train_labels, test_data, test_labels)

    # predictions = mlp.predict(test_data)
    #
    # # This is how you prepare a submission for the competition
    # predictions_csv = {
    #     "ID": [],
    #     "target": [],
    # }
    #
    # for i, label in enumerate(predictions):
    #     predictions_csv["ID"].append(i)
    #     predictions_csv["target"].append(label)
    #
    # df = pd.DataFrame(predictions_csv)
    # df.to_csv("submission.csv", index=False)

