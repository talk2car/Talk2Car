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
