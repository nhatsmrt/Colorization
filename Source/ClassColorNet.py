import numpy as np
import torch
import cv2
from torch import nn
from .NetworkComponent import PretrainedModel, Encoder, Decoder, FusionLayer, ResidualBlock, ConvolutionalLayer, spatial_pyramid_pool
from torch.optim import Adam
import matplotlib.pyplot as plt


# Based on http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf
class ClassColorNet:

    def __init__(self, n_class):
        self._network = _MainNetworkClassColor(n_class)
        self._mse_loss = nn.MSELoss()
        self._class_loss = nn.CrossEntropyLoss()

    def fit(
            self, X, ab, y, X_val = None, ab_val = None,
            batch_size = 16, val_batch_size = 16, n_epoch = 10,
            class_weight = 1, lr = 0.001, print_every = 1, draw = False,
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

        X, ab = (self.__preprocess(X), self.__preprocess(ab))
        y = torch.from_numpy(y).long()


        for e in range(n_epoch):
            print("Epoch " + str(e))
            self._network.train()
            np.random.shuffle(train_indices)

            for i in range(X.shape[0] // batch_size):
                iter_cnt += 1
                start_idx = i * batch_size
                idx = train_indices[start_idx : start_idx + batch_size]

                X_train = X[idx]
                ab_train = ab[idx]
                y_train = y[idx]

                op_ab, class_scores = self._network(X_train)

                optimizer.zero_grad()
                mse_loss = self._mse_loss(op_ab, ab_train)
                class_loss = self._class_loss(class_scores, y_train)
                loss = mse_loss + class_loss * class_weight


                if iter_cnt % print_every == 0:
                    print("Iteration " + str(iter_cnt) + " with mse loss " + str(mse_loss.item()) + " and class loss " + str(class_loss.item()))

                loss.backward()
                optimizer.step()


            # Evaluation:
            if X_val is not None and ab_val is not None:
                print("Evaluating:")
                np.random.shuffle(val_indices)
                val_idx = val_indices[:min(X_val.shape[0], val_batch_size)]
                X_val_batch = X_val[val_idx]
                ab_val_batch = ab_val[val_idx]
                print(ab_val_batch.shape)

                self._network.eval()
                cur_loss = self._mse_loss(
                    self._network(self.__preprocess(X_val_batch))[0],
                    self.__preprocess(ab_val_batch)
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
                        ab_val[random_idx : random_idx + 1]
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
        op = (self._network.colorize(self.__preprocess(X)).permute(0, 2, 3, 1).detach().numpy() + 1) / 2 * 255

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




class _MainNetworkClassColor(nn.Module):
    def __init__(self, n_class):
        super(_MainNetworkClassColor, self).__init__()
        self.add_module("enc_low", _EncoderLow())
        self.add_module("enc_med", _EncoderMedium())
        self.add_module("enc_global", _EncoderGlobal(n_class))
        self.add_module("fusion", FusionLayer(encoded_channels = 32, features_channels = 224))
        self.add_module("dec", Decoder())


    def forward(self, input):
        encoded_low = self._modules["enc_low"](input)
        encoded_medium = self._modules["enc_med"](encoded_low)
        global_features, class_scores = self._modules["enc_global"](encoded_low)

        fused = self._modules["fusion"]((encoded_medium, global_features))

        return (self._modules["dec"](fused, input.shape[2], input.shape[3]), class_scores)

    def colorize(self, input):
        encoded_low = self._modules["enc_low"](input)
        encoded_medium = self._modules["enc_med"](encoded_low)
        global_features = self._modules["enc_global"].global_features(encoded_low)
        fused = self._modules["fusion"]((encoded_medium, global_features))


        return self._modules["dec"](fused, input.shape[2], input.shape[3])




class _EncoderLow(nn.Sequential):
    def __init__(self):
        super(_EncoderLow, self).__init__()
        self.add_module(
            "main",
            nn.Sequential(
                ConvolutionalLayer(
                    in_channels = 1,
                    stride = 2,
                    out_channels = 32
                ),
                ResidualBlock(in_channels = 32),
            )
        )

class _EncoderMedium(nn.Sequential):
    def __init__(self):
        super(_EncoderMedium, self).__init__()
        self.add_module("main", ResidualBlock(in_channels = 32))


class _EncoderGlobal(nn.Module):
    def __init__(self, n_classes):
        super(_EncoderGlobal, self).__init__()
        self.add_module(
            "rep",
            nn.Sequential(
                ConvolutionalLayer(
                    in_channels = 32,
                    stride = 2,
                    out_channels = 16
                ),
                ResidualBlock(in_channels = 16)
            )
        )
        self.add_module(
            "linear",
            nn.Linear(in_features = 224, out_features = n_classes)
        )

    def forward(self, encoded_low):
        rep = self._modules["rep"](encoded_low)
        pooled = spatial_pyramid_pool(
            input = rep,
            op_sizes = {1, 2, 3}
        )
        return (pooled.view(pooled.shape[0], pooled.shape[1], 1, 1), self._modules["linear"](pooled))

    def global_features(self, encoded_low):
        rep = self._modules["rep"](encoded_low)
        pooled = spatial_pyramid_pool(
            input = rep,
            op_sizes = {1, 2, 3}
        )
        return pooled.view(pooled.shape[0], pooled.shape[1], 1, 1)




