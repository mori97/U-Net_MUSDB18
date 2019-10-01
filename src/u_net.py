import torch
import torch.nn.functional as F

EPS = 1e-8


class UNet(torch.nn.Module):
    """An implementation of U-Net for music source separation.
    It has been proposed in "Singing Voice Separation with Deep U-Net
    Convolutional Networks".
    (https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf)

    Args:
        n_class (int): Number of output classes.
    """
    def __init__(self, n_class):
        super(UNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            1, 16, kernel_size=5, stride=2, padding=2)
        self.conv_bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(
            16, 32, kernel_size=5, stride=2, padding=2)
        self.conv_bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(
            32, 64, kernel_size=5, stride=2, padding=2)
        self.conv_bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(
            64, 128, kernel_size=5, stride=2, padding=2)
        self.conv_bn4 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.Conv2d(
            128, 256, kernel_size=5, stride=2, padding=2)
        self.conv_bn5 = torch.nn.BatchNorm2d(256)
        self.conv6 = torch.nn.Conv2d(
            256, 512, kernel_size=5, stride=2, padding=2)
        self.conv_bn6 = torch.nn.BatchNorm2d(512)

        self.deconv1 = torch.nn.ConvTranspose2d(
            512, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn1 = torch.nn.BatchNorm2d(256)
        self.dropout1 = torch.nn.Dropout2d(0.5)
        self.deconv2 = torch.nn.ConvTranspose2d(
            512, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn2 = torch.nn.BatchNorm2d(128)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.deconv3 = torch.nn.ConvTranspose2d(
            256, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn3 = torch.nn.BatchNorm2d(64)
        self.dropout3 = torch.nn.Dropout2d(0.5)
        self.deconv4 = torch.nn.ConvTranspose2d(
            128, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn4 = torch.nn.BatchNorm2d(32)
        self.deconv5 = torch.nn.ConvTranspose2d(
            64, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv_bn5 = torch.nn.BatchNorm2d(16)
        self.deconv6 = torch.nn.ConvTranspose2d(
            32, n_class, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)

        x = torch.log(x + EPS)
        h1 = F.leaky_relu(self.conv_bn1(self.conv1(x)), 0.2)
        h2 = F.leaky_relu(self.conv_bn2(self.conv2(h1)), 0.2)
        h3 = F.leaky_relu(self.conv_bn3(self.conv3(h2)), 0.2)
        h4 = F.leaky_relu(self.conv_bn4(self.conv4(h3)), 0.2)
        h5 = F.leaky_relu(self.conv_bn5(self.conv5(h4)), 0.2)
        h = F.leaky_relu(self.conv_bn6(self.conv6(h5)), 0.2)

        h = self.dropout1(F.relu(self.deconv_bn1(self.deconv1(h))))
        h = torch.cat((h, h5), dim=1)
        h = self.dropout2(F.relu(self.deconv_bn2(self.deconv2(h))))
        h = torch.cat((h, h4), dim=1)
        h = self.dropout3(F.relu(self.deconv_bn3(self.deconv3(h))))
        h = torch.cat((h, h3), dim=1)
        h = F.relu(self.deconv_bn4(self.deconv4(h)))
        h = torch.cat((h, h2), dim=1)
        h = F.relu(self.deconv_bn5(self.deconv5(h)))
        h = torch.cat((h, h1), dim=1)
        h = F.softmax(self.deconv6(h), dim=1)
        return h
