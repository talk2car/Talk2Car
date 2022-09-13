This repository contains the data and development kit for the Talk2Car dataset presented in our paper [Talk2Car: Taking Control of Your Self-Driving Car](https://arxiv.org/pdf/1909.10838).
You can visit the Talk2Car website [here](https://talk2car.github.io).
Project financed by Internal Funds KU Leuven (C14/18/065). Talk2Car is part of the [MACCHINA](https://macchina-ai.cs.kuleuven.be/) project
An example image from Talk2Car given below for the following command: "You can park up ahead behind <b>the silver car, next to that lamppost with the orange sign on it</b>". The referred object is indicated in bold.

<p align="center">
	<img src="static/example.png" />
</p>


# Overview


- [Changelog](#changelog)
- [Talk2Car](#talk2car_overview)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Evaluation](#evaluation)  
  - [Talk2Car Leaderboard](#leaderboard)
- [C4AV Challenge](#c4av_challenge)
  - [C4AV Challenge Quick Start](#c4av_challenge_quick_start)
- [Extensions](#extensions)
- [Citation](#citation)
- [License](#license)


# <a name="changelog"></a>Changelog

- 13th September, 2022: Added Talk2Car-RegSeg dataset in extensions section.
- 26th July, 2022: Added Talk2Car-Destination support in Talk2Car.
- 3rd July, 2022: Added Talk2Car-Destination dataset in the extensions section.
- 1st July, 2022: Added Talk2Car-Expr dataset in the extensions section.
- 29th October, 2021: Added visualization script in utils to create videos up to and including the prediction of the referred object.
- 6th October, 2021: Release of Talk2Car v1.1
- 5th October, 2021: Making the Talk2Car class able to also load the nuScenes dataset or only loading the Talk2Car (Slim dataset version).
- 2th June, 2021: Update code base to pytorch 1.8.1
- 18th March, 2020: First release of Talk2Car.

# <a name="talk2car_overview"></a>Talk2Car

The Talk2Car dataset is built upon the [nuScenes](https://www.nuscenes.org/) dataset. 
Hence, one can use all data provided by nuScenes when using Talk2Car (i.e. LIDAR, RADAR, Video, ...).
However, if one wishes to do so, they need to download 300GB+ of data from nuScenes.
To make our dataset more accessible to researchers and self-driving car enthusiasts with limited hardware,
we also provide a dataset class that can only load the Talk2Car data. For this, one only needs <2GB of storage.
In the following section we first describe the requirements and then we describe how to set up both versions of the dataset.
We also provide a google colab that can train our baseline model found [here](https://colab.research.google.com/drive/19wTtGskfdhZWvNIfIrUcpwe9IChRKBtQ?usp=sharing).

## <a name="requirements"></a> Requirements

To use Talk2Car, we recommend the following instructions:

First, run
```
conda create --name talk2car python=3.6 -y
```

Then
```
source activate talk2car
```

In case you haven't copied the repo yet, run 

```
git clone https://github.com/talk2car/Talk2Car.git
```

Then run:

```
cd Talk2Car && pip install -r requirements.txt
```

Finally,

```
python -m spacy download en_core_web_sm
```


## <a name="setup"></a>Setup

We first start with the dataset version that only uses the Talk2Car data. We will refer to this version as `Talk2CarSlim`.
The version that uses the nuScenes data will be referred to as `Talk2Car`.

### Talk2CarSlim


To set up the data please first follow the requirements section.
Make sure you are at the root directory of Talk2Car for the following instructions.

Activate the `talk2car` environment if this is not done yet.
```
source activate talk2car
```

then,

```
pip install gdown
```

Now download the images

```
gdown --id 1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek
```

Unpack them,

```
unzip imgs.zip && mv imgs/ ./data/images
rm imgs.zip
```

To see if you have installed everything correctly, you should be able to run the following:

```
cd baseline
python3 train.py --root ../data --lr 0.01 --nesterov --evaluate
```

### Talk2Car

First, you need to download the complete nuScenes dataset. 
On their download page, you will need to download all 10 parts.
You will need 300GB+ to download all data. 
The `example.py` file provides an example for loading the data in PyTorch using the Talk2Car class from the `talktocar.py` file. 
We advise to create a conda environment to run the code.
The nuscenes-devkit is required in addition to some popular python packages which can be easily installed through conda or pip. 
The code can be run as follows:

Export the path where you put the nuScenes data and install the nuscenes-devkit through pip.

```
export NUSCENES='/path/to/downloaded/nuscenes'
```

Copy the Talk2Car json files to a directory named 'commands' in the nuScenes dataset folder.

```
export COMMANDS=$NUSCENES'/commands/'
mkdir -p $COMMANDS
cp ./data/commands/* $COMMANDS
```

Run the example file.
```
python3 ./example.py --root $NUSCENES
```

## <a name="evaluation"></a>Evaluation
The object referral task on the Talk2Car dataset requires to predict a bounding box for every command.
The models on Talk2Car are evaluated by checking if the Intersection over Union of the predicted object bounding box and the ground truth bounding box is above 0.5.
This metric can be referred to by many ways i.e.  IoU<sub>0.5</sub>, AP50, ...

If you want to try the evaluation locally on the validation set, you can do so by using `eval.py`.
The script can be used as follows:

```
python eval.py --root $NUSCENES --version val --predictions ./data/predictions.json
```

When replacing the `predictions.json` file by your own model predictions, you are required to follow the same format. 
Specifically, the results need to be stored as a JSON file which contains a python dictionary of the following format {command_token: [x0, y0, w, h]}. Where x0 and y0 are the coordinates of the top left corner, and h, w the height and width of the predicted bounding box.  

Evaluation of your models on the Talk2Car test set is possible [here](https://www.aicrowd.com/challenges/eccv-2020-commands-4-autonomous-vehicles).
An example of this can be seen in the `baseline/test.py`.

## <a name="leaderboard"></a>Leaderboard

The Talk2Car leaderboard can be found [here](leaderboard.md).

# <a name="c4av_challenge"></a>C4AV Challenge

The Talk2Car dataset is part of the [Commands for Autonomous Vehicles](https://www.aicrowd.com/challenges/eccv-2020-commands-4-autonomous-vehicles) challenge.
The challenge required to solve a visual grounding task.
The summary paper of the challenge can be found [here](https://link.springer.com/chapter/10.1007/978-3-030-66096-3_1).

## <a name="c4av_challenge_quick_start"></a>C4AV Challenge - Quick Start

To help participants get started in the C4AV challenge, we provide a [PyTorch code base](https://github.com/talk2car/Talk2Car/tree/master/baseline) that allows to train a baseline model on the Talk2Car dataset within minutes.
Additionally, we include the images and commands as separate files which avoids the need to download the entire nuScenes dataset first. 

# <a name="extensions"></a>Extensions

- Talk2Car-Expr - [Paper](https://www.sciencedirect.com/science/article/pii/S0952197621001044) - [Dataset](https://github.com/ThierryDeruyttere/Talk2Car-Expr)
- Talk2Car-Destination [Paper](https://www.aaai.org/AAAI22Papers/AAAI-8858.GrujicicD.pdf) - [Dataset](https://github.com/ThierryDeruyttere/Talk2Car-Destination)
- Talk2Car-RegSeg [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9636172) - [Dataset](https://rnr-t2c.github.io/)

# <a name="citation"></a>Citation


If you use this work for your own research, please cite:
```
@inproceedings{deruyttere2019talk2car,
  title={Talk2Car: Taking Control of Your Self-Driving Car},
  author={Deruyttere, Thierry and Vandenhende, Simon and Grujicic, Dusan and Van Gool, Luc and Moens, Marie Francine},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2088--2098},
  year={2019}
}
```

# <a name="license"></a>License 


This software is released under an MIT license. For a commercial license please contact the authors.




