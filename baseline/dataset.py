import numpy as np
import os
import json

import torch
import torch.utils.data as data

from PIL import Image
from utils.vocabulary import Vocabulary
from utils.math import jaccard
from talktocar import get_talk2car_class

class Talk2Car(data.Dataset):
    def __init__(self, baseline_data_root, split, talk2car_root, vocabulary='./utils/vocabulary.txt', transform=None):
        self.baseline_data_root = baseline_data_root
        self.split = split

        with open(os.path.join(self.baseline_data_root, 'centernet_bboxes.json'), 'rb') as f:
            data = json.load(f)[self.split] 
            self.data = {int(k): v for k, v in data.items()} # Map to int
        self.img_dir = os.path.join(self.baseline_data_root, 'images')
        self.transform = transform
        self.vocabulary = Vocabulary(vocabulary)
        self.talk2car = get_talk2car_class(root=talk2car_root, split=split, slim=True)

        if self.split in ['val', 'train']:
            self.add_train_annos = True # Add extra info when reading out items for training
        else:
            self.add_train_annos = False

        self.ignore_index = 255 # Ignore index when all RPNs < 0.5 IoU
        self.num_rpns_per_image = 32 # We only use 16 RPN per image

        # Filter out rpns we are not going to use
        # RPNS were obtained from center after soft NMS
        # We order the scores, and take the top k.
        assert(self.num_rpns_per_image < 64)
        rpns = {k: sample['centernet'] for k, sample in self.data.items()}
        rpns_score_ordered_idx = {k: np.argsort([rpn['score'] for rpn in v]) for k, v in rpns.items()}
        rpns = {k: [v[idx] for idx in rpns_score_ordered_idx[k][-self.num_rpns_per_image:]] for k, v in rpns.items()}
        for k in self.data.keys():
            self.data[k]['centernet'] = rpns[k]

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        output = {'index': torch.LongTensor([idx])}
        sample = self.data[idx]
        
        # Load image 
        img_path = os.path.join(self.img_dir, sample['img'])
        
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        output['image'] = img

        # Load command 
        command = self.vocabulary.sent2ix_andpad(sample['command'], add_eos_token=True)
        output['command'] = torch.LongTensor(command)
        output['command_length'] = len(self.vocabulary.sent2ix(sample['command'])) + 1

        # Load region proposals obtained with centernet return bboxes as (xl, yb, xr, yt)
        if 'centernet' in sample.keys():
            # First get all the valid bbox (Remove boxes that are to small)
            bbox = torch.stack([torch.LongTensor(sample['centernet'][i]['bbox']) for i in range(self.num_rpns_per_image)]) # num_rpns x 4
            bbox_lbrt = torch.stack([bbox[:,0], bbox[:,1],
                                     bbox[:,0] + bbox[:,2], 
                                     bbox[:,1] + bbox[:,3]], 1)
            bbox_lbrt[:,0] = torch.clamp(bbox_lbrt[:,0], 0, 1600) # xl
            bbox_lbrt[:,1] = torch.clamp(bbox_lbrt[:,1], 0, 900) # yb
            bbox_lbrt[:,2] = torch.clamp(bbox_lbrt[:,2], 0, 1600) # xr
            bbox_lbrt[:,3] = torch.clamp(bbox_lbrt[:,3], 0, 900) # yt
            output['rpn_bbox_lbrt'] = bbox_lbrt
            
            valid = (bbox_lbrt[:,3] - bbox_lbrt[:,1] > 5) & (bbox_lbrt[:,2] - bbox_lbrt[:,0] > 5)
            
            # Store the region proposals together in one tensor by rescaling them to fixed size
            rpn_image = torch.FloatTensor(self.num_rpns_per_image, 3, 224, 224).zero_()
            for i in range(self.num_rpns_per_image):
                rpn_ = bbox_lbrt[i]
                valid = (rpn_[3] - rpn_[1] > 5) & (rpn_[2] - rpn_[0] > 5)
                if valid:
                    rpn_image[i].copy_(torch.nn.functional.interpolate(img[:, rpn_[1]:rpn_[3], rpn_[0]:rpn_[2]].unsqueeze(0), (224, 224)).squeeze()) 
                else:
                    pass # Will keep zeros
            output['rpn_image'] = rpn_image # Stored as a single tensor

        # Add extra info for training if needed 
        # GT is the proposal with best overlap. 
        # If IoU < 0.5 for best box, add ignore index
        if self.add_train_annos:
            gt = sample['referred_object']
            xl, yl, xt, yt = gt[0], gt[1], gt[0]+gt[2], gt[1]+gt[3]
            output['gt_bbox_lbrt'] = torch.LongTensor([xl,yl,xt,yt])

            iou_array = jaccard(output['rpn_bbox_lbrt'].numpy(), output['gt_bbox_lbrt'].numpy().reshape(1, -1))
            output['rpn_iou'] = torch.from_numpy(iou_array)
            if np.any(iou_array >= 0.5):
                gt = torch.LongTensor([np.argmax(iou_array)]) # Best matching is gt for training
                output['rpn_gt'] = gt
            else:
                output['rpn_gt'] = torch.LongTensor([self.ignore_index]) # No good bbox -> ignore
        else:
            pass
       
        return output 

    def number_of_words(self):
        # Get number of words in the vocabulary
        return self.vocabulary.number_of_words

    def convert_index_to_command_token(self, index):
        return self.data[index]['command_token']

    def convert_command_to_text(self, command):
        # Takes value from command key and transforms it into human readable text
        return ' '.join(self.vocabulary.ix2sent_drop_pad(command.numpy().tolist()))


def main():
    """ A simple example """
    import torchvision.transforms as transforms
    root = '/esat/pyrite/svandenh/datasets/talk2car/'
    split = 'train'
    dataset = Talk2Car(root, split, './utils/vocabulary.txt', transforms.ToTensor())

    print('=> Load a sample')    
    sample = dataset.__getitem__(15)
    img = np.transpose(sample['image'].numpy(), (1,2,0))
    command = dataset.convert_command_to_text(sample['command'])
    print('Command in human readable text: %s' %(command))

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    print('=> Plot image with bounding box around referred object')
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    xl, yb, xr, yt = sample['gt_bbox_lbrt'].tolist()
    w, h = xr - xl, yt - yb
    rect = patches.Rectangle((xl, yb), w, h, fill = False, edgecolor = 'r')
    ax.add_patch(rect)
    plt.axis('off')
    plt.show()

    print('=> Plot image with region proposals (red), gt bbox (blue)')
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i in range(sample['rpn_bbox_lbrt'].size(0)):
        bbox = sample['rpn_bbox_lbrt'][i].tolist()
        xl, yb, xr, yt = bbox 
        w, h = xr - xl, yt - yb
        rect = patches.Rectangle((xl, yb), w, h, fill = False, edgecolor = 'r')
        ax.add_patch(rect)

    gt_box = (sample['rpn_bbox_lbrt'][sample['rpn_gt'].item()]).tolist()
    xl, yb, xr, yt = gt_box
    w, h = xr - xl, yt - yb
    rect = patches.Rectangle((xl, yb), w, h, fill = False, edgecolor = 'b')
    ax.add_patch(rect)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('bboxes.png', bbox_inches='tight')
 
if __name__=='__main__':
    main()
