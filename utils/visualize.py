"""
Script to visualize referred object predictions in a video.
The created videos are from the first frame of the scene until the referred object prediction is made.
"""

from talk2car import get_talk2car_class
import argparse
import json
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--data_root")
parser.add_argument('--split', default="val")
parser.add_argument("--command_path")
parser.add_argument("--visualize_json")
parser.add_argument("--freq", default=10)
parser.add_argument("--thickness", default=3)

args = parser.parse_args()

# If you want to visualize samples, make sure you have the full nuScenes dataset!
t2c = get_talk2car_class(args.data_root, args.split, args.command_path, slim=False)
command_token_to_command = {x.command_token: x for x in t2c.commands}

### Visualize json should be of dictionary of format
### {"command_token": [x,y,w,h], ...}
to_visualize = json.load(open(args.visualize_json, "r"))

im_size = (640, 360)
color = (0,0,255) #Red in BGR format

for command_token, (x,y,w,h) in to_visualize.items():
    # Get the command class obj
    command = command_token_to_command[command_token]
    scene_token = command.scene_token
    scene_obj = t2c.get("scene", scene_token)

    # Get first token
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(command_token+".mp4", fourcc, args.freq, im_size)

    next_token = scene_obj["first_sample_token"]
    curr_sample = t2c.get("sample", next_token)
    fc_rec = curr_sample["data"]["CAM_FRONT"]

    # Get records from DB
    sd_rec = t2c.get("sample_data", fc_rec)
    has_more_frames = True

    while has_more_frames:
        impath, _, _ = t2c.get_sample_data(sd_rec["token"])
        im = cv2.imread(impath)

        if sd_rec["token"] == command.frame_token:
            # Draw box
            start_point = (x,y)
            end_point = (x+w, y+h)
            im = cv2.rectangle(im, start_point, end_point, color, args.thickness)
            im = cv2.resize(im, im_size)
            out.write(im)
            break
        im = cv2.resize(im, im_size)
        out.write(im)

        if not sd_rec['next'] == "":
            sd_rec = t2c.get('sample_data', sd_rec['next'])
        else:
            has_more_frames = False

    out.release()
