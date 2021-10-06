# Dataset

The Talk2Car commands are stored as JSON files. These can easily be loaded using the following python commands:

```
import json
with open('train.json', 'r') as f:
	train = json.load(f)
```

The data is split into a train, val and test set. The splits are stored as python dictionaries with two keys, i.e. scene tokens and commands. The scene tokens (list of strings) contain the nuScenes scene tokens of the scenes that are contained in the split. The train and validation commands are stored as a list of dictionaries of the following format:

```
{
    'scene_token': f92422ed4b4e427194a4958ccf15709a, # nuScenes scene token
    'sample_token': c32d636e44604d77a1734386b3fe4a0d, # nuScenes sample token
    'translation': [-13.49250542687401, 0.43033061594724364, 59.28095610405408], # Translation
    'size': [0.81, 0.73, 1.959], # Size
    'rotation':  ['-0.38666213835670615', '-0.38076281276237284', '-0.5922192111910205', '0.5956412318459762'], # Rotation,
    'command': 'turn left to pick up the pedestrian at the corner', # Command
    'obj_name': 'human.pedestrian.adult', # Class name of the reffered object 
    'box_token': '0183ed8a474f411f8a3394eb78df7838' # nuScenes box token,
    'command_token': '4175173f5f60d19ecfc3712e960a1103' # A unique command identifier,
    '2d_box': [200, 300, 50, 50] # The 2d bounding box of the referred object in the frontal view. Follows the format [x,y,w,h]
    '':
}
```

The test commands are stored as a list of dictionaries of the following format:

```
{
    'scene_token': f92422ed4b4e427194a4958ccf15709a, # nuScenes scene token
    'sample_token': c32d636e44604d77a1734386b3fe4a0d, # nuScenes sample token
    'command': 'turn left to pick up the pedestrian at the corner', # Command
    'command_token': '4175173f5f60d19ecfc3712e960a1103' # A unique command identifier
}
```
