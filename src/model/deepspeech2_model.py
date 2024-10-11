import torch
from torch import nn
from torch.nn import Sequential, BatchNorm1d

class GRUWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUWithBatchNorm, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.batchnorm = BatchNorm1d(hidden_size)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)
        T, N = x.shape[0], x.shape[1] 
        x = x.view(T * N, -1)
        x = self.batchnorm(x) 
        # [T, N, hidden_size]
        x = x.view(T, N, -1)
        return x


class DeepSpeech2Model(nn.Module):
    def __init__(self, n_feats, fc_hidden, n_tokens):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.conv = Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), padding=(20, 5), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), padding=(10, 5), stride=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(21, 11), padding=(10, 5), stride=(2, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        input_size = ((((((n_feats + 40 - 41) // 2 + 1) + 20 - 21) // 2 + 1) + 20 - 21) // 2 + 1) * 96

        self.grus = nn.ModuleList([GRUWithBatchNorm(input_size, fc_hidden)] + 
                                  [GRUWithBatchNorm(fc_hidden, fc_hidden) for _ in range(4)])

        self.fc = nn.Linear(fc_hidden, n_tokens)
        self.paddings = [(20, 5), (10, 5), (10, 5)]
        self.kernel_sizes = [(41, 11), (21, 11), (21, 11)]
        self.strides = [(2, 2), (2, 1), (2, 1)]

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = self.conv(spectrogram.unsqueeze(1))  # [N, 1, F, T]
        N, C, F, T = x.shape
        x = x.view(N, C * F, T)  # [N, C*F, T]
        x = x.transpose(1, 2)  # [N, T, C*F]
        x = x.transpose(0, 1).contiguous()  # [T, N, C*F]

        for gru in self.grus:
            x = gru(x)

        T, N = x.shape[0], x.shape[1]
        x = x.view(T * N, -1) 
        x = self.fc(x)
        x = x.view(T, N, -1)
        x = x.transpose(0, 1)  # [N, T, n_tokens]

        return {'log_probs': nn.functional.log_softmax(x, dim=-1), 'log_probs_length': self.transform_input_lengths(spectrogram_length)}

    def transform_input_lengths(self, input_lengths):
        for i in range(3):
            input_lengths = torch.floor((input_lengths + 2 * self.paddings[i][1] - (self.kernel_sizes[i][1] - 1) - 1) / self.strides[i][1] + 1).to(torch.int)
        return input_lengths

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
