import json
import argparse
import numpy as np
from talktocar import Talk2Car

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument("--root", help="root")
FLAGS.add_argument("--version", help="version")
FLAGS.add_argument("--predictions", help="JSON file with predictions")


def evaluate(ground_truth, predictions):
    """
    args: 
    -ground_truth: A dictionary with command_tokens as keys, and the predicted ground truth boxes
    as values. The values are lists of the format [x0, y0, h, w].
    -predictions: Should have same structure as the ground_truth. 
    """
    assert(set(ground_truth.keys()) == set(predictions.keys()))

    # Result
    result = {}

    # Calculate IoU for every bounding box
    iou_list = []
    for cmd_token in set(ground_truth.keys()):
        gt, pred = ground_truth[cmd_token], predictions[cmd_token]
        gt_x0_y0_x1_y1 = np.array([gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]).reshape(
            1, -1
        )
        pred_x0_y0_x1_y1 = np.array(
            [pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3]]
        ).reshape(1, -1)
        iou_list.append(jaccard(gt_x0_y0_x1_y1, pred_x0_y0_x1_y1).squeeze().item())

    # Mean IoU
    print("Mean Iou is %.4f" % (np.mean(iou_list)))
    result["Mean IoU"] = np.mean(iou_list)

    # Average precision
    threshold = np.arange(0.5, 1.0, 0.05).tolist()
    AP_list = []
    for thresh in threshold:
        iou_thresholded = [int(iou > thresh) for iou in iou_list]
        AP_list.append(np.mean(iou_thresholded))
        result["AP%d" % (int(100 * thresh))] = np.mean(iou_thresholded)
        print("AP%d is %.4f" % (int(100 * thresh), np.mean(iou_thresholded)))

    print("AP50..95 is %.4f" % (np.mean(AP_list)))
    result["AP50..95"] = np.mean(AP_list)
    return result


def jaccard(a, b):
    # pairwise jaccard(IoU) botween boxes a and boxes b
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    inter = np.clip(rb - lt, 0, None)

    area_i = np.prod(inter, axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)

    area_u = area_a[:, np.newaxis] + area_b - area_i
    return area_i / np.clip(area_u, 1e-7, None)  # len(a) x len(b)


def main():
    args = FLAGS.parse_args()

    # Ground truth
    print('Gather ground truth')
    dataset = Talk2Car(version=args.version, dataroot=args.root, verbose=False)
    ground_truth = {}
    for i in range(len(dataset.commands)):
        command = dataset.commands[i]
        bbox = list(map(int, command.get_2d_bbox()))  # x0, y0, h, w
        cmd_hash = command.get_command_token() # string
        ground_truth[cmd_hash] = bbox
    
    # Predictions
    print('Read predictions from JSON file %s' %(args.predictions))
    with open(args.predictions, "r") as f:
        predictions = json.load(f)
    
    # Evaluate
    print('Perform evaluation')
    evaluate(ground_truth, predictions)

main()
