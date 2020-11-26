import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# custom module
from util import *
from src.dataset import *
from src.model import EfficientDet


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")


def get_args():
    ''' get arguments '''
    parser = argparse.ArgumentParser("EfficientDet")
    parser.add_argument("--expname", type=str,
                        default='EXP', help="experiment name")
    parser.add_argument("--best", type=str,
                        default="/HW2model.pth.tar", help="best model")

    args = parser.parse_args()
    args.best = args.expname + args.best

    return args


def main(args):
    torch.cuda.manual_seed(1)

    # load test data
    test_params = {"batch_size": 1,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": 2}

    test_set = TestDataset()
    test_loader = DataLoader(test_set, **test_params)

    # define model
    model = EfficientDet(num_classes=10)
    model = model.to(device)
    model = nn.DataParallel(model)

    # load model
    print('loading checkpoint {}'.format(args.best))
    checkpoint = torch.load(args.best)
    args.best_loss = checkpoint['best_loss']
    model.load_state_dict(checkpoint['state_dict'])
    print('loaded checkpoint {}'.format(args.best))
    print('best loss:', args.best_loss)

    # test
    test(model, test_loader, args)


def test(model, test_loader, args):
    ''' predict and write json file '''
    model.eval()
    bbox = []
    label = []
    score = []

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            print(i, '/', len(test_loader), end='     \r')
            image = sample['img']
            scale = sample['scale']
            scale = float(scale)

            image = image.to(device).float()

            # predict
            nms_scores, nms_class, nms_anchors = model([image])

            # rescale the bounding boxes to original size
            nms_scores = nms_scores.cpu().detach().numpy()
            nms_class = nms_class.cpu().detach().numpy()
            nms_anchors = (nms_anchors / scale).cpu().detach().numpy()

            bbox.append(nms_anchors)
            label.append(nms_class)
            score.append(nms_scores)

    # write json file
    with open("prediction.json", 'w') as File:
        File.write('[')
        for i in range(13068):
            bbs = [list(np.around(b).astype(np.int16)) for b in bbox[i]]
            bbox[i] = [[bb[1], bb[0], bb[3], bb[2]] for bb in bbs]
            label[i] = [b if b else 10 for b in label[i]]
            score[i] = [b for b in score[i]]

            string = ('{"bbox": ' + str(bbox[i]) +
                      ', "label": ' + str(label[i]) +
                      ', "score": ' + str(score[i]) +
                      '}')
            File.write(string)
            if i < 13067:
                File.write(', \n')
        File.write(']')

    return


if __name__ == "__main__":
    args = get_args()
    main(args)
