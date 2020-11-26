import torch
import torch.nn as nn
import numpy as np


class BBoxTransform(nn.Module):
    ''' Transform Anchors to Prediction
        Bounding Boxes
    '''

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()

    def forward(self, boxes, deltas):
        # get w, h, x, y of anchors
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        # get prediction bounding boxes'
        # w, h, x, y
        dx = deltas[:, :, 0] * 0.1
        dy = deltas[:, :, 1] * 0.1
        dw = deltas[:, :, 2] * 0.2
        dh = deltas[:, :, 3] * 0.2
        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        # get prediction bounding boxes'
        # top, left, bottom, right
        pred_x1 = pred_ctr_x - 0.5 * pred_w
        pred_y1 = pred_ctr_y - 0.5 * pred_h
        pred_x2 = pred_ctr_x + 0.5 * pred_w
        pred_y2 = pred_ctr_y + 0.5 * pred_h

        return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=2)


class ClipBoxes(nn.Module):
    ''' Clip Bounding Boxes '''

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        _, _, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class Anchors(nn.Module):
    ''' Module for Generating Anchors '''

    def __init__(self):
        super(Anchors, self).__init__()
        # feature maps' level
        self.pyramid_levels = [1, 2, 3, 4]
        # stride between two anchors
        self.strides = [2 ** x for x in self.pyramid_levels]
        # anchor's size
        self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        # ratio of anchors (h / w)
        self.ratios = np.array([2.5, 1.25])
        # anchor's scales
        self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x)
                        for x in self.pyramid_levels]

        # generate anchors and append them
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(self.sizes[idx],
                                       self.ratios,
                                       self.scales)
            shifted_anchors = shift(image_shapes[idx],
                                    self.strides[idx],
                                    anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        # to torch tensor
        anchors = torch.from_numpy(all_anchors.astype(np.float32))
        anchors = anchors.cuda()
        return anchors


def generate_anchors(base_size, ratios, scales):
    ''' generate anchors in different ratio and scale '''
    num_anchors = len(ratios) * len(scales)
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    areas = anchors[:, 2] * anchors[:, 3]
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    ''' Shift generated anchors to obtain anchors
    in other positions '''
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors
