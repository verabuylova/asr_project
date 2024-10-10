import torch
from torch import nn
from torch.nn import Sequential, BatchNorm1d

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
            nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (41, 11), padding = (20, 5), stride = (2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (21, 11), padding = (10, 5), stride = (2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels = 32, out_channels = 96, kernel_size = (21, 11), padding = (10, 5), stride = (2, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        input_size = ((((((n_feats + 40 - 41) // 2 + 1) + 20 - 21) // 2 + 1) + 20 - 21) // 2 + 1) * 96

        self.grus = nn.ModuleList([nn.GRU(input_size, fc_hidden, bidirectional = True)] + 
                                  [nn.GRU(fc_hidden, fc_hidden, bidirectional = True) for _ in range(4)])
        
        self.batchnorms = nn.ModuleList([BatchNorm1d(fc_hidden) for _ in range(5)])

        self.fc = nn.Linear(fc_hidden, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = self.conv(spectrogram.unsqueeze(1))
        N, C, F, T = x.shape
        x = x.view(N, C * F, T)
        x = x.transpose(1, 2) 
        x = x.transpose(0, 1).contiguous()

        for gru, batchnorm in zip(self.grus, self.batchnorms):
            x, h = gru(x)
            x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)
            T_gru, N_gru = x.shape[0], x.shape[1]
            x = x.view((T_gru * N_gru, -1))
            x = batchnorm(x)
            x = x.view((T_gru, N_gru, -1)).contiguous()

        T, N = x.shape[0], x.shape[1]
        x = x.view(T * N, -1) 
        x = self.fc(x)
        x = x.view(T, N, -1)
        x = x.transpose(0, 1)
        
        return {'log_probs': nn.functional.log_softmax(x, dim = -1), 'log_probs_length': self.transform_input_lengths(spectrogram_length)}

    def transform_input_lengths(self, input_lengths):
        return torch.zeros_like(input_lengths).fill_((((((input_lengths.max() + 2 * 5 - 11) // 2 + 1) + 2 * 5 - 11) // 2 + 1) + 2 * 5 - 11) + 1)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info