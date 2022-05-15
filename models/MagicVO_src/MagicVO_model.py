import torch
from torch import nn

class MagicVO_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        # Recurrent Neural Netwok:
        self.hidden_size = 1000
        self.num_layers = 2
        self.input_size = 10 * 3 * 1024
        self.bi_lstm_model = torch.nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, \
            bidirectional=True, dropout=0.5).to(self.device)
        # Fully Connected Layer:
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size*2, 256),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Linear(256, 6) # Note that 6 corresponds to 6 DoF predictions
        ).to(self.device)

    def forward(self, x):
        # Flatten the input:
        print(x.shape, "MagicVO input shape")
        out = x.view(-1, self.input_size) # --> [B, L, 10*3*1024]
        print(out.shape, "MagicVO input flattened")
        out = out.unsqueeze(0)
        
        # Pass through Recurrent Layers:
        batch_size = 1
        h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size).to(self.device)
        out, _ = self.bi_lstm_model(out, (h0, c0)) # --> [B, L, hidden_size]
        # Pass through Dense Layers:
        out = self.fc(out) # --> [B, L, 6]
        out = out.squeeze(0) # --> [L, 6] # Note that B = 1
        print(out.shape, " MagicVO output shape")
        return out

    def loss(self, x, targets, k):
        preds = self.forward(x) # --> [L, 6] 6 DoF predictions
        preds = preds.float()
        targets = targets.float()
        mse_position_loss = torch.nn.functional.mse_loss(preds[:, :3], targets[:, :3])
        mse_orientation_loss = torch.nn.functional.mse_loss(preds[:, 3:], targets[:, 3:])
        combined_loss = mse_position_loss + (k * mse_orientation_loss)
        return combined_loss

