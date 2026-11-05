import torch.nn as nn


class EmbedV2V(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_dim, latent_dim // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(latent_dim // 2, latent_dim, kernel_size=3, padding=1)

        self._initialize_weights()
        
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                # nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
