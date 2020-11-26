import torch
import torch.nn as nn
import os


def calc_iou(a, b):
    ''' Calculate IoUs of anchors and Bounding Boxes '''
    GT = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    PRED = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)

    # width of intersection
    right = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2])
    left = torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    iw = right - left

    # height of intersection
    top = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3])
    bottom = torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    ih = top - bottom

    # area of intersection
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih

    # IoU
    ua = PRED + GT - intersection
    ua = torch.clamp(ua, min=1e-8)
    IoU = intersection / ua
    return IoU


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations):
        ''' Compute Loss
        - classifications: every blocks with prediction
        - regressions: every blocks with offset and scale
        - anchors: original blocks
        - annotations: GT with class and bbox
        '''
        # set hyperparameters of focal loss
        alpha = 0.25
        gamma = 5.

        # Initialize Parameters
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        anchors = anchors[0]
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

        # For each images, calculate it's losses
        for j in range(batch_size):
            # define predictions and ground truth
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if bbox_annotation.shape[0] == 0:
                # If no annotations, skip it
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            # clamp the scores to 1e-8 to 1 - 1e-8
            # to prevent from nan/inf loss
            classification = torch.clamp(classification, 1e-8, 1.0 - 1e-8)

            # Calculate every anchors' most similar bbox annotations
            IoU = calc_iou(anchors, bbox_annotation[:, :4])
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # *******************************************
            # *** compute the loss for classification ***
            # *******************************************
            targets = torch.ones(classification.shape).cuda() * -1
            # IoU < 0.4 Back Ground
            targets[torch.lt(IoU_max, 0.4), :] = 0

            # IoU > 0.5 -> fore ground
            # assigned_annotations: class
            pos_idx = torch.ge(IoU_max, 0.5)
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            targets[pos_idx, :] = 0
            targets[pos_idx,
                    assigned_annotations[pos_idx, 4].long()] = 1

            # calculate Focal Weight
            alpha_factor = torch.where(torch.eq(targets, 1.), 1., alpha)
            focal_weight = torch.where(torch.eq(targets, 1.),
                                       1. - classification,
                                       classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            # classification loss = mean(binaryCE * Focalweight)
            bce = -(targets * torch.log(classification) +
                    (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce
            zeros = torch.zeros(cls_loss.shape).cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            num_positive_anchors = pos_idx.sum()
            num_p = torch.clamp(num_positive_anchors.float(), min=1.0)
            classification_losses.append(cls_loss.sum() / num_p)

            # ***************************************
            # *** compute the loss for regression ***
            # ***************************************
            if num_positive_anchors > 0:
                # transform annotations to deltas (dx, dy, dw, dh)
                assigned_annotations = assigned_annotations[pos_idx]

                anchor_widths_pi = anchor_widths[pos_idx]
                anchor_heights_pi = anchor_heights[pos_idx]
                anchor_ctr_x_pi = anchor_ctr_x[pos_idx]
                anchor_ctr_y_pi = anchor_ctr_y[pos_idx]

                gt_right = assigned_annotations[:, 2]
                gt_left = assigned_annotations[:, 0]
                gt_widths = gt_right - gt_left

                gt_top = assigned_annotations[:, 3]
                gt_bottom = assigned_annotations[:, 1]
                gt_heights = gt_top - gt_bottom
                gt_ctr_x = gt_left + 0.5 * gt_widths
                gt_ctr_y = gt_bottom + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                # stack it up and normalize it
                targets = torch.stack(
                    (targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                norm = torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                targets = targets / norm

                regression_diff = torch.abs(targets - regression[pos_idx, :])

                # smooth_l1
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean().cuda())
            else:
                # if no positive anchors, reg loss = 0.
                regression_losses.append(torch.tensor(0).float().cuda())

        return (torch.stack(classification_losses).mean(dim=0, keepdim=True),
                torch.stack(regression_losses).mean(dim=0, keepdim=True))
