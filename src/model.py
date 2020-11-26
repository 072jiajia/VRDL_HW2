import math
import torch
import torch.nn as nn
from torchvision.ops.boxes import nms as nms_torch
from torchvision import models
from torch.nn import functional as F

# costum module
from src.utils import *
from src.loss import FocalLoss


def nms(dets, thresh):
    ''' Obtain the Bounding Box of Prediction
    - dets: rectangles whose score is above a confidence threshold
    - thresh: threshold of NMS algorithm

    algorithm:
    (1) add the rectangle (a) with the highest score to my prediction
    (2) compute IoUs between the rectangle (a) and the others
    (3) remove the rectangles whose IoU is over 'thresh'
    (4) to (1) util all rectangles are removed or added into prediction
    '''
    anchors = dets[:, :4]
    scores = dets[:, 4]
    return nms_torch(anchors.cpu(), scores.cpu(), thresh)


class ConvBlock(nn.Module):
    ''' Convolution block for BiFPN (above)
    A convolution layer with efficiency
    '''

    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3,
                      stride=1, padding=1, groups=num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=num_channels,
                           momentum=0.999, eps=1e-5),
            nn.LeakyReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class BiFPN(nn.Module):
    ''' Bi-Directional Feature Pyramid Network
    applied in EfficientDet
    '''

    def __init__(self, num_channels):
        super(BiFPN, self).__init__()
        # Conv layers
        self.conv3_up = ConvBlock(num_channels)
        self.conv2_up = ConvBlock(num_channels)
        self.conv1_up = ConvBlock(num_channels)
        self.conv2_down = ConvBlock(num_channels)
        self.conv3_down = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.Softmax = nn.Softmax(dim=0)

        # Weight
        self.p3_w1 = nn.Parameter(torch.ones(2))
        self.p2_w1 = nn.Parameter(torch.ones(2))
        self.p1_w1 = nn.Parameter(torch.ones(2))

        self.p2_w2 = nn.Parameter(torch.ones(3))
        self.p3_w2 = nn.Parameter(torch.ones(3))
        self.p4_w2 = nn.Parameter(torch.ones(2))

    def forward(self, inputs):
        """
            P4_0 -------------------------- P4_2 -------->
                              v               ^
                ______________v____________   ^
              /               v             \ ^
            P3_0 ---------- P3_1 ---------- P3_2 -------->
                              v               ^
                ______________v____________   ^
              /               v             \ ^
            P2_0 ---------- P2_1 ---------- P2_2 -------->
                              v               ^
                              v               ^
            P1_0 -------------------------- P1_2 -------->
        """

        p1_in, p2_in, p3_in, p4_in = inputs

        # higher scale feature to lower scale feature
        p3_w1 = self.Softmax(self.p3_w1)
        w = p3_w1 / (torch.sum(p3_w1, dim=0))
        p3_up = self.conv3_up(w[0] * p3_in + w[1] * self.upsample(p4_in))

        p2_w1 = self.Softmax(self.p2_w1)
        w = p2_w1 / (torch.sum(p2_w1, dim=0))
        p2_up = self.conv2_up(w[0] * p2_in + w[1] * self.upsample(p3_up))

        p1_w1 = self.Softmax(self.p1_w1)
        w = p1_w1 / (torch.sum(p1_w1, dim=0))
        p1_out = self.conv1_up(w[0] * p1_in + w[1] * self.upsample(p2_up))

        # higher scale feature to lower scale feature
        p2_w2 = self.Softmax(self.p2_w2)
        w = p2_w2 / (torch.sum(p2_w2, dim=0))
        p2_out = self.conv2_down(
            w[0] * p2_in + w[1] * p2_up + w[2] * self.downsample(p1_out))

        p3_w2 = self.Softmax(self.p3_w2)
        w = p3_w2 / (torch.sum(p3_w2, dim=0))
        p3_out = self.conv3_down(
            w[0] * p3_in + w[1] * p3_up + w[2] * self.downsample(p2_out))

        p4_w2 = self.Softmax(self.p4_w2)
        w = p4_w2 / (torch.sum(p4_w2, dim=0))
        p4_out = self.conv4_down(w[0] * p4_in + w[1] * self.downsample(p3_out))

        return p1_out, p2_out, p3_out, p4_out


class Regressor(nn.Module):
    ''' Regressor to obtain the offset and the rescaling
    factor of anchors
    '''

    def __init__(self, in_channels, num_anchors, num_layers):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * 4,
                                kernel_size=3, padding=1)

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        output = inputs.permute(0, 2, 3, 1)
        return output.contiguous().view(output.shape[0], -1, 4)


class Classifier(nn.Module):
    ''' Classifier to obtain the score of classes
    in anchors
    '''

    def __init__(self, in_channels, num_anchors, num_classes, num_layers):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels,
                                    kernel_size=3, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * num_classes,
                                kernel_size=3, padding=1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        inputs = self.act(inputs)
        inputs = inputs.permute(0, 2, 3, 1)
        output = inputs.contiguous().view(inputs.shape[0],
                                          inputs.shape[1],
                                          inputs.shape[2],
                                          self.num_anchors,
                                          self.num_classes)
        return output.contiguous().view(output.shape[0],
                                        -1,
                                        self.num_classes)


class ResNet(nn.Module):
    ''' ResNet
        I use the first few layers of pre-trained ResNet
    as my backbone because about most of images in the dataset
    have size smaller than 100 pixels
    '''

    def __init__(self):
        super(ResNet, self).__init__()
        model = models.resnet34(pretrained=True)
        del model.fc
        del model.layer4
        self.model = model

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x1 = x = self.model.relu(x)
        x = self.model.maxpool(x)

        x2 = x = self.model.layer1(x)
        x3 = x = self.model.layer2(x)
        x4 = x = self.model.layer3(x)

        return [x1, x2, x3, x4]


class EfficientDet(nn.Module):
    ''' My Smaller EfficientDet '''

    def __init__(self, num_anchors=6, num_classes=10):
        super(EfficientDet, self).__init__()
        # Backbone
        self.backbone_net = ResNet()

        # define the number of channel in feature pyramid
        num_channels = 64

        # Convs to obtain the features in feature pyramid
        self.conv1 = nn.Conv2d(64, num_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(64, num_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(128, num_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(256, num_channels, kernel_size=1)

        # bi-directional feature pyramid network
        self.bifpn = nn.Sequential(*[BiFPN(num_channels) for _ in range(2)])

        # define regressor and classifier
        self.regressor = Regressor(in_channels=num_channels,
                                   num_anchors=num_anchors,
                                   num_layers=2)
        self.classifier = Classifier(in_channels=num_channels,
                                     num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=2)

        # Module for generating anchors
        self.anchors = Anchors()

        # module for transforming anchor boxes into
        # Mrediction bounding box
        self.regressBoxes = BBoxTransform()

        # Module for cliping edges
        self.clipBoxes = ClipBoxes()

        # Module of Loss Function
        self.focalLoss = FocalLoss()

        # Initialize Weights and Bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(
            -math.log((1.0 - prior) / prior))

        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)

    def forward(self, inputs):
        ''' Forwarding Function
        - Training: Get 2 Inputs, one is for images and
                another is for annotations
            Output the Loss directly
        - Testing:
            Input is one image
            Return the prediction bounding boxes, classes, scores
        '''
        if len(inputs) == 2:
            is_training = True
            img_batch, annotations = inputs
        else:
            is_training = False
            img_batch = inputs[0]

        # Obtain the feature maps extract their features
        # in feature pyramid
        c1, c2, c3, c4 = self.backbone_net(img_batch)
        # print(c1.shape, c2.shape, c3.shape, c4.shape)
        p1 = self.conv1(c1)
        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)

        # bi-directional feature extraction
        features = [p1, p2, p3, p4]
        features = self.bifpn(features)

        # Generate anchors for this image
        anchors = self.anchors(img_batch)

        # Obtain the offsets, rescaling factors and scores
        regression = torch.cat([self.regressor(feature)
                                for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature)
                                    for feature in features], dim=1)

        if is_training:
            # If is training, return cls loss and reg loss
            return self.focalLoss(classification, regression,
                                  anchors, annotations)
        else:
            # transform anchors to prediction bounding box
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(
                transformed_anchors, img_batch)

            # get the bounding boxes with threshold > 0.2
            # if the number of bounding boxes is not over 500,
            # threshold *= 0.5 util there's over 500 bounding boxes
            scores = torch.max(classification, dim=2, keepdim=True)[0]

            thresh = 0.2
            scores_over_thresh = (scores > thresh)[0, :, 0]

            while scores_over_thresh.sum() < 500:
                thresh *= 0.5
                scores_over_thresh = (scores > thresh)[0, :, 0]

            # get the bounding boxes' position, class, confidence(score)
            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            # use non-maximum suppressioin to drop out similar boxes
            scored_anchors = torch.cat([transformed_anchors, scores], dim=2)
            nms_idx = nms(scored_anchors[0, :, :], 0.5)
            nms_scores, nms_class = classification[0, nms_idx, :].max(dim=1)
            return [nms_scores, nms_class, transformed_anchors[0, nms_idx, :]]
