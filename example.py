""" This file contains an example of a dataloader for the Talk2Car dataset """
import argparse
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset
from vocabulary import Vocabulary
from talktocar import get_talk2car_class

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument("--root", help="Dataset root")

# Arguments
args = FLAGS.parse_args()

# Train set
version = "train"

# Dataroot
root = args.root

# Vocabulary
vocabulary = Vocabulary("t2c_voc.txt")

# An example dataset implementation
class ExampleDataset(Dataset):
    def __init__(self, root, vocabulary, command_path=None, version="train", slim_t2c=True):
        # Initialize
        self.version = version
        self.vocabulary = vocabulary
        self.dataset = get_talk2car_class(root, split=self.version, slim=slim_t2c, command_path=command_path)

        # Fixed params
        self.num_classes = 23
        self.class_names = [
            "animal",
            "human.pedestrian.adult",
            "human.pedestrian.child",
            "human.pedestrian.construction_worker",
            "human.pedestrian.personal_mobility",
            "human.pedestrian.police_officer",
            "human.pedestrian.stroller",
            "human.pedestrian.wheelchair",
            "movable_object.barrier",
            "movable_object.debris",
            "movable_object.pushable_pullable",
            "movable_object.trafficcone",
            "static_object.bicycle_rack",
            "vehicle.bicycle",
            "vehicle.bus.bendy",
            "vehicle.bus.rigid",
            "vehicle.car",
            "vehicle.construction",
            "vehicle.emergency.ambulance",
            "vehicle.emergency.police",
            "vehicle.motorcycle",
            "vehicle.trailer",
            "vehicle.truck",
        ]
        self.class_name_to_id = dict(map(reversed, enumerate(self.class_names)))
        self.id_to_class_name = dict(enumerate(self.class_names))

    def __len__(self):
        return len(self.dataset.commands)

    def __getitem__(self, index):
        # Command
        command = self.dataset.commands[index]
        descr = torch.Tensor(
            self.vocabulary.sent2ix_andpad(command.command, add_eos_token=True)
        ).long()
        length = len(self.vocabulary.sent2ix(command.command)) + 1

        # Get paths
        sd_rec = self.dataset.get("sample_data", command.frame_token)
        img_path, _, _ = self.dataset.get_sample_data(sd_rec["token"])

        # Img
        img = np.array(imageio.imread(img_path), dtype=np.float32)
        img = img / 255.0
        img = torch.from_numpy(img).float()

        # Bbox [x0,y0,w,h]
        bbox = np.array(list(map(int, command.get_2d_bbox())))

        return img, descr, torch.LongTensor([length]), bbox


# Retrieve dataset
dataset = ExampleDataset(root, vocabulary, version)
print("Number of %s samples is %d" % (version, dataset.__len__()))

# Visualize one sample
sample = dataset.__getitem__(0)
print("Text: %s" % (dataset.vocabulary.ix2sent_drop_pad(sample[1].numpy().tolist())))
print("Text contains %d words" % (sample[2].item()))
fig, ax = plt.subplots(1)
ax.imshow(sample[0].numpy())
x0, y0, w, h = (
    sample[3][0].item(),
    sample[3][1].item(),
    sample[3][2].item(),
    sample[3][3].item(),
)
rect = patches.Rectangle((x0, y0), w, h, fill=False, edgecolor="r")
ax.add_patch(rect)
plt.show()
