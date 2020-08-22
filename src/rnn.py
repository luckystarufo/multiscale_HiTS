import torch
import torch.nn as nn


class LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMWrapper, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, t=1):
        # x is of shape batch_size x seq_len x feature
        # h0 and c0 default to zero
        batch_size, seq_len, _ = x.shape
        y_preds = torch.zeros(batch_size, seq_len + t - 1, self.input_size)

        x, (h, c) = self.rnn(x)
        x = x.contiguous().view(batch_size * seq_len, self.hidden_size)
        y = self.fc(x).view(batch_size, seq_len, self.input_size)
        y_preds[:, :seq_len, :] = y

        # extra
        for i in range(t-1):
            y_pred = y[:, [-1], :]
            x_pred, (h, c) = self.rnn(y_pred, (h, c))
            x_pred = x_pred.contiguous().view(batch_size, self.hidden_size)
            y = self.fc(x_pred).view(batch_size, 1, self.input_size)
            y_preds[:, seq_len+i, :] = y

        return y_preds

