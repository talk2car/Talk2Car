import argparse
import json

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms

from dataset import Talk2Car
from utils.collate import custom_collate

import models.resnet as resnet
import models.nlp_models as nlp_models

parser = argparse.ArgumentParser(description='Talk2Car object referral')
parser.add_argument('--root', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=18, type=int,
                    metavar='N',
                    help='mini-batch size (default: 18)')

def main():
    args = parser.parse_args()

    # Create dataset
    print("=> creating dataset")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    dataset = Talk2Car(talk2car_root=args.root, split='test',
                                transform=transforms.Compose([transforms.ToTensor(), normalize]))
    dataloader = data.DataLoader(dataset, batch_size = args.batch_size, shuffle=True,
                            num_workers=args.workers, collate_fn=custom_collate, pin_memory=True,                            drop_last=False)
    print('Test set contains %d samples' %(len(dataset)))

    # Create model
    print("=> creating model")
    img_encoder = resnet.__dict__['resnet18'](pretrained=True) 
    text_encoder = nlp_models.TextEncoder(input_dim=dataset.number_of_words(),
                                                 hidden_size=512, dropout=0.1)
    img_encoder.cuda()
    text_encoder.cuda()

    cudnn.benchmark = True

    # Evaluate model
    print("=> Evaluating best model")
    checkpoint = torch.load('best_model.pth.tar', map_location='cpu')
    img_encoder.load_state_dict(checkpoint['img_encoder'])
    text_encoder.load_state_dict(checkpoint['text_encoder'])
    evaluate(dataloader, img_encoder, text_encoder, args)

@torch.no_grad()
def evaluate(val_dataloader, img_encoder, text_encoder, args):
    img_encoder.eval()
    text_encoder.eval()
    
    ignore_index = val_dataloader.dataset.ignore_index
    prediction_dict = {}       

    for i, batch in enumerate(val_dataloader):

        # Data
        region_proposals = batch['rpn_image'].cuda(non_blocking=True)
        command = batch['command'].cuda(non_blocking=True)
        command_length = batch['command_length'].cuda(non_blocking=True)
        b, r, c, h, w = region_proposals.size()
        if len(batch["index"].shape) == 0:
            region_proposals = region_proposals.unsqueeze(0)
            command = command.unsqueeze(0)
            command_length = command_length.unsqueeze(0)

        # Image features
        img_features = img_encoder(region_proposals.view(b*r, c, h, w))
        norm = img_features.norm(p=2, dim=1, keepdim=True)
        img_features = img_features.div(norm)

        # Sentence features
        _, sentence_features = text_encoder(command.permute(1,0), command_length)
        norm = sentence_features.norm(p=2, dim=1, keepdim=True)
        sentence_features = sentence_features.div(norm)

        # Product in latent space
        scores = torch.bmm(img_features.view(b, r, -1), sentence_features.unsqueeze(2)).squeeze()

        if len(batch["index"].shape) == 0:
            scores = scores.unsqueeze(0)

        pred = torch.argmax(scores, 1)

        # Add predictions to dict
        for i_, idx_ in enumerate(batch['index'].tolist()):
            token = val_dataloader.dataset.convert_index_to_command_token(idx_)
            bbox = batch['rpn_bbox_lbrt'][i_, pred[i_]].tolist()
            bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
            if token in prediction_dict.keys():
                print('Token already exists')
            prediction_dict[token] = bbox


    print('Predictions for %d samples saved' %(len(prediction_dict)))
    with open('predictions.json', 'w') as f:
        json.dump(prediction_dict, f)
        
if __name__ == "__main__":
    main()
