import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50


def jaccard_loss(inputs, targets, mean_reduce=True):
    inputs = inputs.view(inputs.shape[0], -1)
    targets = targets.view(inputs.shape[0], -1)

    numer = 2 * inputs * targets
    denom = inputs + targets

    loss = torch.sum(numer, dim=1) / (1e-6 + torch.sum(denom, dim=1))

    if mean_reduce:
        loss = torch.mean(loss)

    return loss


def my_loss(inputs, targets, mean_reduce=True):
    inputs = inputs.view(inputs.shape[0], -1)
    targets = targets.view(inputs.shape[0], -1)

    numer = inputs * targets
    denom = inputs

    loss = 1 - torch.sum(numer, dim=1) / (1e-6 + torch.sum(denom, dim=1))

    if mean_reduce:
        loss = torch.mean(loss)

    return loss


class Placer(nn.Module):

    def __init__(self):
        super(Placer, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, (7, 7), stride=(2, 2), padding=3), nn.MaxPool2d(2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 128, (5, 5), stride=(2, 2), padding=2), nn.MaxPool2d(2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), stride=(2, 2), padding=2), nn.MaxPool2d(2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), stride=(2, 2), padding=2), nn.MaxPool2d(2, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 9)
        )

    def compute_params(self, x_rect, y_rect):
        x = torch.cat([x_rect, y_rect], dim=1)
        output = self.model(x)

        return output

    def forward(self, x_rect, y_rect, x_means=None):
        x = torch.cat([x_rect, y_rect], dim=1)
        output = self.model(x)

        rot_part = torch.stack([torch.stack([torch.cos(output[:, :(output.shape[1] // 3)]),
                                             -torch.sin(output[:, :(output.shape[1] // 3)])], dim=2),
                                torch.stack([torch.sin(output[:, :(output.shape[1] // 3)]),
                                             torch.cos(output[:, :(output.shape[1] // 3)])], dim=2)],
                               dim=2)

        shift = output[:, (output.shape[1] // 3):].reshape(output.shape[0], -1, 2)

        if x_means is not None:
            shift = x_means - torch.einsum('bijk, bik -> bij', rot_part, x_means - shift)

        theta = torch.cat([rot_part, shift.view(*shift.shape, 1)], dim=3)

        theta = theta.reshape(-1, 2, 3)
        x = x_rect.reshape(-1, 1, *x_rect.shape[2:])

        grid = F.affine_grid(theta, x.shape)
        y = F.grid_sample(x, grid)

        x_means = torch.cat([x_means, torch.ones(*x_means.shape[:2], 1)], dim=2)
        x_means = x_means.view(-1, *x_means.shape[2:])

        x_means = torch.einsum('bij, bj -> bi', theta, x_means)

        x_means = x_means.view(x_rect.shape[0], -1, x_means.shape[1])

        return y.reshape(x_rect.shape), x_means


class PlacerResnet(nn.Module):

    def __init__(self):
        super(PlacerResnet, self).__init__()

        self.model = resnet50()
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(2048, 9)

    def compute_params(self, x_rect, y_rect):
        x = torch.cat([x_rect, y_rect], dim=1)
        output = self.model(x)

        return output

    def forward(self, x_rect, y_rect, x_means=None):
        x = torch.cat([x_rect, y_rect], dim=1)
        output = self.model(x)

        rot_part = torch.stack([torch.stack([torch.cos(output[:, :(output.shape[1] // 3)]),
                                             -torch.sin(output[:, :(output.shape[1] // 3)])], dim=2),
                                torch.stack([torch.sin(output[:, :(output.shape[1] // 3)]),
                                             torch.cos(output[:, :(output.shape[1] // 3)])], dim=2)],
                               dim=2)

        shift = output[:, (output.shape[1] // 3):].reshape(output.shape[0], -1, 2)

        if x_means is not None:
            shift = x_means - torch.einsum('bijk, bik -> bij', rot_part, x_means - shift)

        theta = torch.cat([rot_part, shift.view(*shift.shape, 1)], dim=3)

        theta = theta.reshape(-1, 2, 3)
        x = x_rect.reshape(-1, 1, *x_rect.shape[2:])

        grid = F.affine_grid(theta, x.shape)
        y = F.grid_sample(x, grid)

        x_means = torch.cat([x_means, torch.ones(*x_means.shape[:2], 1, device=x_means.device)], dim=2)
        x_means = x_means.view(-1, *x_means.shape[2:])

        x_means = torch.einsum('bij, bj -> bi', theta, x_means)

        x_means = x_means.view(x_rect.shape[0], -1, x_means.shape[1])

        return y.reshape(x_rect.shape), x_means


class PlacerPL(pl.LightningModule):

    def __init__(self):
        super(PlacerPL, self).__init__()

        self.model = PlacerResnet()

    def training_step(self, batch, batch_idx):
        x_rect, y_rect, x_means = batch

        x_rect_tr, x_means = self.model(x_rect, y_rect, x_means)

        pos_part = sum(my_loss(x_rect_tr[:, k: k + 1], y_rect, mean_reduce=False) for k in range(x_rect.shape[1]))
        neg_part = sum(jaccard_loss(x_rect_tr[:, i], x_rect_tr[:, j], mean_reduce=False)
                       for i in range(x_rect.shape[1]) for j in range(i + 1, x_rect.shape[1]) if
                       i != j)

        total_loss = pos_part + neg_part

        self.log('pos_loss', pos_part.mean())
        self.log('neg_loss', neg_part.mean())
        self.log('total_loss', total_loss.mean())

        return total_loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10,
                                                               min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "total_loss"}
