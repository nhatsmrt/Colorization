import numpy as np
import torch
import cv2
from torch import nn
from .NetworkComponent import ResizeConvolutionalLayer, Encoder
from torch.optim import Adam
import matplotlib.pyplot as plt

class SimpleColorNet:

    def __init__(self):
        self._network = _MainNetwork()
        self._loss = nn.CrossEntropyLoss()

    def fit(
            self, X, y, X_val = None, y_val = None,
            batch_size = 16, n_epoch = 10,
            lr = 0.001,
            print_every = 1, draw = False,
            save_path = None, load_path = None, patience = 3
    ):
        train_indices = np.arange(X.shape[0])
        optimizer = Adam(self._network.parameters(), lr = lr)
        iter_cnt = 0

        if load_path is not None:
            self.load_weight(load_path)

        min_loss = None
        p = 0

        X = self.__preprocess(X)
        y = self.__preprocess_ab(y)


        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._network.train()
            np.random.shuffle(train_indices)

            for i in range(X.shape[0] // batch_size):
                iter_cnt += 1
                start_idx = i * batch_size
                idx = train_indices[start_idx : start_idx + batch_size]

                # X_train = self.__preprocess(X[idx])
                # y_train = self.__preprocess_ab(y[idx])

                X_train = X[idx]
                y_train = y[idx]

                op = self._network(X_train)

                optimizer.zero_grad()
                loss = self._loss(op, y_train)
                if iter_cnt % print_every == 0:
                    print("Iteration " + str(iter_cnt) + " with loss " + str(loss.item()))

                loss.backward()
                optimizer.step()


            # Evaluation:
            if X_val is not None and y_val is not None:
                print("Evaluating:")
                cur_loss = self._loss(
                    self._network(self.__preprocess(X_val)),
                    self.__preprocess_ab(y_val)
                ).item()

                print("Val loss: " + str(cur_loss))

                # Early stopping and weight saving
                if (min_loss is None or cur_loss < min_loss):
                    print("Val loss decrease.")
                    min_loss = cur_loss
                    p = 0

                    if save_path is not None:
                        self.save_weight(save_path)
                else:
                    p += 1

                if draw:
                    random_idx = np.random.choice(X_val.shape[0])

                    val_img_BGR = self.colorize(X_val[random_idx : random_idx + 1])[0]
                    true_img_lab = self.__postprocess(
                        X_val[random_idx : random_idx + 1],
                        y_val[random_idx : random_idx + 1]
                    )[0]

                    val_img = cv2.cvtColor(val_img_BGR, cv2.COLOR_BGR2RGB)
                    true_img = cv2.cvtColor(true_img_lab, cv2.COLOR_LAB2RGB)


                    fig = plt.figure()
                    fig.add_subplot(1, 2, 1)

                    plt.imshow(val_img)

                    fig = plt.figure()
                    fig.add_subplot(1, 2, 2)
                    plt.imshow(true_img)

                    plt.show()

                if p > patience:
                    print("Patience exceeded. Training finished.")
                    return

    # Input: X (Grayscale), with same dimensions.
    # Return colorized X in BGR color
    def colorize(self, X):
        if (len(X.shape) != 4 or X.shape[3] != 1):
            raise ValueError

        self._network.eval()
        op = np.argmax(self._network(self.__preprocess(X)).permute(0, 3, 4, 2, 1).detach().numpy(), -1)

        lab = self.__postprocess(X, op)


        ret = []
        for img in lab:
            ret.append(cv2.cvtColor(img, cv2.COLOR_LAB2BGR))

        return np.array(ret)


    def save_weight(self, PATH):
        torch.save(self._network.state_dict(), PATH + "/network.pt")
        print("Weight saved successfully")


    def load_weight(self, PATH):
        self._network.load_state_dict(torch.load(PATH + "/network.pt"))
        print("Weight loaded successfully")



    def __preprocess(self, X):
        return torch.from_numpy(X / 255 * 2 - 1).permute(0, 3, 1, 2).float()

    def __preprocess_ab(self, ab):
        return torch.from_numpy(ab).permute(0, 3, 1, 2).long()

    def __postprocess(self, X, y):
        return np.concatenate((X, y), axis = -1).astype(np.uint8)


class _MainNetwork(nn.Module):

    def __init__(self):
        super(_MainNetwork, self).__init__()
        self._enc = Encoder()
        self._upsample = ResizeConvolutionalLayer(in_channels = 16, out_channels = 512)
        self._softmax = nn.Softmax(dim = 1)


    def forward(self, X):
        rep = self._enc(X)
        upsampled = self._upsample(rep, X.shape[2], X.shape[3])
        upsampled_reshape = upsampled.view(upsampled.shape[0], 256, 2, upsampled.shape[2], upsampled.shape[3])

        return upsampled_reshape