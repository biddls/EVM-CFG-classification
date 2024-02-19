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
        print(f"Size of data: {len(data)}")
        self.data, self.weights = self.findWeights(data, CFGs)
        del data # its alot so, good to free up

        self.data = [torch.Tensor(x).to(device) for x in tqdm(self.data, desc="Loading data to GPU")]

        width = self.data[0].shape[1]

        self.model = LSTMAE(width, embedding_dim).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.criterion = nn.MSELoss()

    def trainEnc(self, epochs_num: int):
        # todo: losses are going up fix
        """
        Input shape:
        [samples(fixed), steps (varies), features(fixed)]
        """
        self.epochs_num = epochs_num
        if len(self.data) == 0:
            raise ValueError("No training data provided")

        train_loss=0
        # loop = tqdm(
        #     range(self.epochs_num),
        #     desc=f"TL: nan",
        #     position=0,
        #     leave=True,
        #     unit="Epoch"
        # )
        
        for epoch in range(self.epochs_num):
            # Training mode
            self.model.train()
            self.optimizer.zero_grad()
            train_losses = list()
            loopTrain = tqdm(
                zip(self.data, self.weights),
                leave=False,
                desc=f"Epoch {epoch+1}/{self.epochs_num}",
                position=1,
                total=len(self.data),
                ncols=0,
                mininterval=0.1
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
            print(f'\nLoss: {str(train_loss)[:6]} | {now.strftime("%H:%M %d/%m/%y")}', end='')

            if epoch % 10 == 0 and epoch != 0:
                # saves the model every 10 epochs
                torch.save(self.model.state_dict(), f"./src/vectorEncoding/cache/checkpointsLSTMAutoenc/LSTM_Autoenc{epoch}of{self.epochs_num}.pt")

    def findWeights(self, data: Counter[tuple[int | tuple[int, int]]], CFGs: int) -> tuple[list[npt.NDArray[np.bool_]], list[float]]:
        """
        Finds the weights of the data
        This is like a basic frequency count
        """

        freq = list(data.values())
        
        print(f"Number of training examples: {np.sum(freq):,.0f}")
        print(f"Number now compressed: {len(freq):,.0f}")
        print(f"Compression ratio of: {100 * (1 - (len(freq) / np.sum(freq))):.2f}%")
        
        freq = np.array(freq) / CFGs
        # add e to every element
        freq = freq + np.e - 1 + (1 - (1 / CFGs))
        # take the natural log of every element
        freq = np.log(freq)
        # conver to a list
        freq = freq.tolist()
        freq = [float(x) for x in freq]

        temp_data = list(data.keys())
        temp_data = map(lambda x : Tokeniser.vectoriseNode(list(x)), temp_data)
        temp_data = list(temp_data)

        return temp_data, freq

    def countSample(self, sample1: npt.NDArray[np.bool_], data: list[npt.NDArray[np.bool_]]) -> int:
        """
        Checks if the sample is valid
        """
        count = 0
        for sample in data:
            if sample1.shape != sample.shape:
                continue
            elif np.array_equal(sample1, sample):
                count += 1
        return count

    def findSample(self, sample1: npt.NDArray[np.bool_], data: list[npt.NDArray[np.bool_]]) -> bool:
        """
        Checks if the sample is valid
        """
        for sample in data:
            if sample1.shape != sample.shape:
                continue
            elif np.array_equal(sample1, sample):
                return True
        return False


if __name__ == "__main__":
    # [samples, steps, features]
    data = np.random.rand(10, 20, 30)
    trainer = LSTM_AutoEnc_Training(data, 32, 10) # type: ignore
    trainer.trainEnc(100)
