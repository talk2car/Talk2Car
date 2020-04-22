# Talk2Car

This repository contains the data and development kit for the Talk2Car dataset presented in our paper [Talk2Car: Taking Control of Your Self-Driving Car](https://arxiv.org/pdf/1909.10838).
You can visit the Talk2Car website [here](https://talk2car.github.io). 

If you use this work for your own research, please consider citing:
```
@inproceedings{deruyttere2019talk2car,
  title={Talk2Car: Taking Control of Your Self-Driving Car},
  author={Deruyttere, Thierry and Vandenhende, Simon and Grujicic, Dusan and Van Gool, Luc and Moens, Marie Francine},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2088--2098},
  year={2019}
}
```

## C4AV Challenge

The Talk2Car dataset is part of the [Commands for Autonomous Vehicles](https://c4av-2020.github.io/) (C4AV) challenge. The challenge requires to solve a visual grounding task. The following sections introduce the dataset, some example images and an evaluation script. The format required to submit to the challenge is the same as the one used in the evaluation example. More details about the visual grounding task on Talk2Car are provided in the paper. Finally, we include a simple baseline as a python notebook to help people get acquainted with the task.  

## C4AV Challenge - Quick Start

To help participants get started in the C4AV challenge, we provide a PyTorch code base [click here](https://github.com/talk2car/Talk2Car/tree/master/c4av_model) that allows to train a baseline model on the Talk2Car dataset within minutes. Additionally, we include the images and commands as separate files which avoids the need to download the entire nuScenes dataset first. 

## Dataset

The Talk2Car dataset is built upon the [nuScenes](https://www.nuscenes.org/) dataset. The nuScenes dataset has to be downloaded separately. The `example.py` file provides an example for loading the data in PyTorch using the Talk2Car class from the `talktocar.py` file. We advise to create a conda environment to run the code. The nuscenes-devkit is required in addition to some popular python packages which can be easily installed through conda or pip. The code can be run as follows:

Export the path where you put the nuScenes data and install the nuscenes-devkit through pip.

```
export NUSCENES='/usr/data/nuscenes/'
pip install nuscenes-devkit 
```

Copy the Talk2Car json files to a directory named 'commands' in the nuScenes dataset folder.

```
mkdir talk2car
cd talk2car
git clone https://github.com/commands-selfdriving-car/Talk2Car-dataset.git .
export COMMANDS=$NUSCENES'/commands/'
mkdir -p $COMMANDS
cp ./data/* $COMMANDS
```

Run the example file.
```
python3 ./example.py --root $NUSCENES
```

## Example
An example image is given below for the following command: "You can park up ahead behind <b>the silver car, next to that lamppost with the orange sign on it</b>". The referred object is indicated in bold.

<p align="center">
	<img src="static/example.png" />
</p>

## Evaluation
The object referral task on the Talk2Car dataset requires to predict a bounding box for every command. We use AP(>0.5) as the main performance metric. The evaluation script is implemented in `eval.py`. The script can be used as follows:

```
python eval.py --root $NUSCENES --version val --predictions ./data/predictions.json
```

When replacing the `predictions.json` file by your own model predictions, you are required to follow the same format. Specifically, the results need to be stored as a JSON file which contains a python dictionary of the following format {command_token: [x0, y0, h,w]}. Where x0 and y0 are the coordinates of the top left corner, and h, w the height and width of the predicted bounding box.  

## License 

This software is released under an MIT license. For a commercial license please contact the authors.




