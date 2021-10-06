# C4AV Challenge

This directory contains a code base to help researchers participate in the [C4AV Challenge](https://www.aicrowd.com/challenges/eccv-2020-commands-4-autonomous-vehicles). A model is trained for the object referral task by matching an embedding of the command with an embedding of object region proposals obtained with CenterNet. A technical report detailing the solution can be found [here](https://arxiv.org/abs/2004.13822).

The images are delivered as a single zip file, together with a json file that contains the necessary annotations and bounding box coordinates of region proposals extracted with CenterNet.

## Notebook

If you're interested in a google colab notebook with this baseline, you can find it here: https://colab.research.google.com/drive/19wTtGskfdhZWvNIfIrUcpwe9IChRKBtQ?usp=sharing

## Setup and First Run
If you just want a script that downloads the repository, the data and the required packages, please copy the following 

```
# Create conda environment
conda create --name talk2car_baseline python=3.6 -y
echo "created conda environment 'talk2car_baseline'..."

source activate talk2car_baseline
echo "activating environment 'talk2car_baseline'..."

echo "installing 'gdown'..."
pip install gdown

# Download code
echo "downloading the code..."
git clone https://github.com/talk2car/Talk2Car.git

# Download data
echo "downloading the data..."
gdown --id 1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek

# Unzip images to the correct directory
echo "preparing the data..."
unzip imgs.zip && mv imgs/ Talk2Car/c4av_model/data/images
rm imgs.zip

# Install requirements
echo "installing talk2car requirements..."
cd Talk2Car/c4av_model && pip install -r requirements.txt

echo "installing spacy's en_core_web_sm..."
python -m spacy download en_core_web_sm

echo "start training..."
python3 train.py --root ./data --lr 0.01 --nesterov --evaluate
```

in a shell script i.e. 'setup_and_run.sh' in a new directory where you want to download everything.
After copying the code in the file, run `chmod +x setup_and_run.sh && ./setup_and_run.sh`

## Minified Dataset
The images can be found [here](https://drive.google.com/open?id=1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek). Unzip the images and copy them to the ./data folder. Notice that the ./data folder contains two separate json files. In one case, we list 64 possibly overlapping region proposals extracted by CenterNet. In the latter case, we removed duplicate proposals from the list.

You can run the code as follows.

```
python3 train.py --root ./data --lr 0.01 --nesterov --evaluate 
```

The published code can be used to train a model that obtains +- 42% AP50 on the validation set. The training can be done on a single 1080ti GPU in just a few hours.

## Submission
A submission file can be created by running the test.py script. This will create a predictions.json file that can be uploaded to the our test server on [AICrowd](https://www.aicrowd.com/challenges/eccv-2020-commands-4-autonomous-vehicles). The predictions.json file that is delivered with the repository contains the predictions of the pre-trained model. 

```
python3 test.py --root ./data
```
 
## Requirements

See requirements.txt for a list of packages. Notice that you need to install the English language model for spacy by running the following command from within your environment.

```
python -m spacy download en_core_web_sm
```

## Pretrained models

Obtain a pretrained model [here](https://drive.google.com/open?id=1-FsTYjMxv7-Pw_eXHyDOGTgDlscRyA1j).

