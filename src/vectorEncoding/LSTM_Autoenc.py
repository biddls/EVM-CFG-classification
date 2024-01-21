import torch
import torch.nn as nn

device = torch.device('cuda')


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=32):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
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
        x, (_, _) = self.rnn1(x)
        _, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((x.shape[0], self.embedding_dim))

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, input_dim=32):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features
        self.rnn1 = nn.LSTM(
        input_size=input_dim,
        hidden_size=input_dim,
        num_layers=1,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=input_dim,
        hidden_size=self.hidden_dim,
        num_layers=1,
        batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        return self.output_layer(x)


class LSTMAE(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=32):
        super(LSTMAE, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, n_features, embedding_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def getVectors(self, x):
        return self.encoder.getVectors(x)

class training:
    def __init__(self, data, epochs_num):
        train_size = int(0.8 * len(data))
        self.train, self.valid = data[:train_size], data[train_size:]

        self.epochs_num = epochs_num
        self.model = LSTMAE(data.shape[1], data[2].shape[1], embedding_dim=32).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.criterion = nn.MSELoss()

    def trainEnc(self):
        for epoch in range(self.epochs_num):
            # Training mode
            self.model.train()
            self.optimizer.zero_grad()

            seq_pred = self.model(self.train.to(device))
            loss = self.criterion(seq_pred, self.train)
            loss.backward()
            self.optimizer.step()
            train_loss = loss.item()

            # Evaluation mode
            self.model.eval()
            with torch.no_grad():
                seq_pred_valid = self.model(self.valid.to(device))
                val_loss = self.criterion(seq_pred_valid, self.valid).item()

            print(f'Epoch [{epoch+1}/{self.epochs_num}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


if __name__ == "__main__":
    data = torch.rand((10, 20, 32)).to(device)
    trainer = training(data, 100)
    trainer.trainEnc()
    out = trainer.model.getVectors(data)
