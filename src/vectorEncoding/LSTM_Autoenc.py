import os
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import datetime as dt
from collections import Counter
from tokeniser import Tokeniser

"""
Save:
    torch.save(model.state_dict(), PATH)
Load:
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
"""

device = torch.device('cuda')

class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        # [len of smaple, features]
        # [20, 30]
        x, (_, _) = self.rnn1(x)
        # [len of smaple, embedding_dim]
        # [20, 64]
        _, (hidden, _) = self.rnn2(x)
        # [64]
        # hidden.shape => [1, batch_size, embedding_dim]
        return hidden.reshape(-1)


class Decoder(nn.Module):
    def __init__(self, n_features, input_dim):
        super(Decoder, self).__init__()
        self.input_dim = n_features
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x, sample_len):
        # [32]
        # stack the vector to get back the timesteps
        x = x.repeat(sample_len, 1)
        # [20, 32]
        x, (_, _) = self.rnn1(x)
        # [20, 64]
        x, (_, _) = self.rnn2(x)
        # [20, 64]
        # it runs the output layer on each time step independently
        # and returns the result
        return self.output_layer(x)


class LSTMAE(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super(LSTMAE, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim).to(device)
        self.decoder = Decoder(n_features, embedding_dim).to(device)

    def forward(self, x):
        sample_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, sample_len)
        return x


class LSTM_AutoEnc_Training:
    def __init__(self, data: Counter[tuple[int | tuple[int, int]]], embedding_dim: int, CFGs: int):
        # print(f"Size of data: {len(data)}")
        self.embedding_dim = embedding_dim
        
        path = f"./src/vectorEncoding/cache/checkpointsLSTMAutoenc/width{self.embedding_dim}/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.data, self.weights = self.findWeights(data, CFGs)
        del data # its alot so, good to free up

        self.data = [torch.Tensor(x).to(device) for x in self.data]

        width = self.data[0].shape[1]

        self.model = LSTMAE(width, embedding_dim).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.criterion = nn.MSELoss()


    def trainEnc(self, epochs_num: int, checkpoints: bool = False, progress: bool = False) -> list[float]:
        """
        Input shape:
        [samples(fixed), steps (varies), features(fixed)]
        returns the final loss
        """
        if len(self.data) == 0:
            raise ValueError("No training data provided")

        # todo: convert to batched training

        width = len(str(epochs_num))
        train_loss=0
        final_train_losses = list()
        # Training mode
        self.model.train()
        history = 20
        for epoch in range(1, epochs_num):
            self.optimizer.zero_grad()
            train_losses = list()

            loopTrain = tqdm(
                zip(self.data, self.weights),
                leave=False,
                desc=f"Epoch {epoch: {width}d}/{epochs_num}",
                position=1,
                total=len(self.data),
                ncols=0,
                mininterval=0.5,
                unit="samples"
            )
            # runs through each sample
            for sample, weight in loopTrain:
                seq_pred = self.model(sample)
                # reverses the sequence
                seq_pred = seq_pred.flip(0)

                train_loss = self.criterion(seq_pred, sample)
                train_loss = train_loss * weight
                train_loss.backward()
                self.optimizer.step()
                train_losses.append(train_loss.item())

            train_loss = np.mean(train_losses)
            # loop.set_description(f'TL: {str(train_loss)[:6]}')
            now = dt.datetime.now()
            # width -= len(str(epoch+1))
            if progress:
                print(f'\nEpoch{epoch: {width+1}d}/{epochs_num}| Loss: {train_loss:.8f} | {now.strftime("%H:%M:%S %d/%m/%y")}', end='')

            # check for early stopping
            if len(final_train_losses) > history:
                backAv = sum(final_train_losses[-history:-1])/(history-1)
                if backAv * .99 < final_train_losses[-1]:
                    torch.save(self.model.state_dict(), f"./src/vectorEncoding/cache/checkpointsLSTMAutoenc/width{self.embedding_dim}/LSTM_Autoenc_FINAL{epoch}of{epochs_num}.pt")
                    print(f"Early stopping at epoch {epoch}")
                    # print(f"{backAv * .99 = }")
                    # print(f"{final_train_losses[-1] = }")
                    break

            # checkpointing
            if (epoch % 5 == 0 and epoch != 1) and checkpoints:
                # saves the model every 10 epochs
                torch.save(self.model.state_dict(), f"./src/vectorEncoding/cache/checkpointsLSTMAutoenc/width{self.embedding_dim}/LSTM_Autoenc{epoch}of{epochs_num}.pt")
            final_train_losses.append(float(train_loss))

        return final_train_losses


    def findWeights(self, data: Counter[tuple[int | tuple[int, int]]], CFGs: int) -> tuple[list[npt.NDArray[np.bool_]], list[float]]:
        """
        Finds the weights of the data
        This is like a basic frequency count
        """

        weights = list(data.values())
        # todo think about how i want to handle weights
        # print(f"Number of training examples: {np.sum(weights):,.0f}")
        # print(f"Number now compressed: {len(weights):,.0f}")
        # print(f"Compression ratio of: {100 * (1 - (len(weights) / np.sum(weights))):.2f}%")
        
        weights = np.array(weights) / CFGs
        # add e to every element
        weights = weights + np.e - 1 + (1 - (1 / CFGs))
        # take the natural log of every element
        weights = np.log(weights)
        # conver to a list
        weights = weights.tolist()
        weights = [float(x) for x in weights]

        temp_data = list(data.keys())
        temp_data = map(lambda x : Tokeniser.vectoriseNode(list(x)), temp_data)
        temp_data = list(temp_data)

        return temp_data, list(np.ones_like(weights))


    def getEncodings(self) -> npt.NDArray[np.float64]:
        """
        Returns the encodings of the data
        """
        self.model.eval()
        encodings = list()
        for sample in tqdm(self.data, desc="generating encodings"):
            encodings.append(self.model.encoder(sample).cpu().detach().numpy())
        return np.array(encodings)
