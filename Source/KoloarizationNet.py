import numpy as np
import torch
import cv2
from torch import nn
from .NetworkComponent import PretrainedModel, Encoder, Decoder, FusionLayer
from torch.optim import Adam
import matplotlib.pyplot as plt


# Based on https://arxiv.org/pdf/1712.03400.pdf
class KoalarizationNet:

    def __init__(self):
        self._network = _MainNetworkKoala()
        self._loss = nn.MSELoss()

    def fit(
            self, X, y, X_val = None, y_val = None,
            batch_size = 16, val_batch_size = 16, n_epoch = 10,
            lr = 0.001, print_every = 1, draw = False,
            save_path = None, load_path = None, patience = 3
    ):
        train_indices = np.arange(X.shape[0])
        val_indices = np.arange(X_val.shape[0])

        optimizer = Adam(self._network.parameters(), lr = lr)
        iter_cnt = 0

        if load_path is not None:
            self.load_weight(load_path)

        min_loss = None
        p = 0

        X, y = (self.__preprocess(X), self.__preprocess(y))


        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._network.train()
            np.random.shuffle(train_indices)

            for i in range(X.shape[0] // batch_size):
                iter_cnt += 1
                start_idx = i * batch_size
                idx = train_indices[start_idx : start_idx + batch_size]

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
                np.random.shuffle(val_indices)
                val_idx = val_indices[:min(X_val.shape[0], val_batch_size)]
                X_val_batch = X_val[val_idx]
                y_val_batch = y_val[val_idx]

                self._network.eval()
                cur_loss = self._loss(
                    self._network(self.__preprocess(X_val_batch)),
                    self.__preprocess(y_val_batch)
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
        op = (self._network(self.__preprocess(X)).permute(0, 2, 3, 1).detach().numpy() + 1) / 2 * 255

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



    # def __preprocess_ab(self, ab):
    #     return torch.from_numpy(ab).permute(0, 3, 1, 2).long()

    def __postprocess(self, X, y):
        return np.concatenate((X, y), axis = -1).astype(np.uint8)




class _MainNetworkKoala(nn.Module):
    def __init__(self):
        super(_MainNetworkKoala, self).__init__()
        self.add_module("enc", Encoder())
        self._pre = PretrainedModel(fine_tune = False)
        self.add_module("fusion", FusionLayer())
        self.add_module("dec", Decoder())


    def forward(self, input):
        encoded = self._modules["enc"](input)
        features = self._pre(self.__pretrained_preprocess(input))[0]
        fused = self._modules["fusion"]((encoded, features))

        return self._modules["dec"](fused, input.shape[2], input.shape[3])

    def __pretrained_preprocess(self, input):
        return input.repeat(1, 3, 1, 1)


