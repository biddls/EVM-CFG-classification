import torch
import torch.nn as nn
import numpy as np

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
            batch_first=True
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        return x

    def getVectors(self, x):
        """remeber to feed in for each node"""
        raise NotImplementedError
        x, (_, _) = self.rnn1(x)
        _, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((x.shape[0], self.embedding_dim))

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

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        # it runs the output layer on each time step independently
        # and returns the result
        return self.output_layer(x)


class LSTMAE(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super(LSTMAE, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim).to(device)
        self.decoder = Decoder(n_features, embedding_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def getVectors(self, x):
        """remeber to feed in for each node"""
        return self.encoder.getVectors(x)


class LSTM_AutoEnc_Training:
    def __init__(self, data, embedding_dim):
        train_size = int(0.8 * len(data))
        self.train, self.valid = data[:train_size], data[train_size:]

        self.train = torch.tensor(self.train).float().to(device)
        self.valid = torch.tensor(self.valid).float().to(device)

        if isinstance(data, list):
            width = data[0].shape[0]
        else:
            width = data.shape[2]

        self.model = LSTMAE(width, embedding_dim).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.criterion = nn.MSELoss()

    def trainEnc(self, epochs_num):
        """
        Input shape:
        [samples, steps, features]
        """
        self.epochs_num = epochs_num
        for epoch in range(self.epochs_num):
            # Training mode
            self.model.train()
            self.optimizer.zero_grad()

            seq_pred = self.model(self.train)
            loss = self.criterion(seq_pred, self.train)
            loss.backward()
            self.optimizer.step()
            train_loss = loss.item()

            if epoch % 10 == 0:
                # Evaluation mode
                self.model.eval()
                with torch.no_grad():
                    seq_pred_valid = self.model(self.valid)
                    val_loss = self.criterion(seq_pred_valid, self.valid).item()

                print(f'Epoch [{epoch+1}/{self.epochs_num}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


if __name__ == "__main__":
    # [samples, steps, features]
    data = np.random.rand(10, 20, 30)
    trainer = LSTM_AutoEnc_Training(data, 32)
    trainer.trainEnc(100)
    out = trainer.model.getVectors(data)
