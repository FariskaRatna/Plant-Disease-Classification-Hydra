import torch
from torch import nn
from efficientnet_pytorch import EfficientNet

class EfficientNetCustom(nn.Module):
    def __init__(
        self,
        num_classes,
        model_name="efficientnet-b0",
        dropout_rate=0.2,
        lin1_size=128,
        lin2_size=64,
    ) -> None:
        super().__init__()

        self.efficientnet = EfficientNet.from_pretrained(model_name)

        self.additional_layers = nn.Sequential(
            nn.Conv2d(1280, lin1_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate)
        )

        in_features = self.efficientnet._fc.in_features
        self.additional_fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(lin1_size, lin2_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(lin2_size, num_classes),
        )
        self.efficientnet._fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x = self.efficientnet.extract_features(x)

        x = self.additional_layers(x)

        x = x.mean([2, 3])

        x = self.additional_fc(x)

        return x


if __name__ == "__main__":
    _ = EfficientNet()
