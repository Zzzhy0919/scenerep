# scenerep
Installation
1. Create environment
conda create -n scenerep python=3.8
conda activate scenerep

2. Install requirements
pip install -r requirements.txt

3. Clone OWL-ViT (big_vision)
git clone https://github.com/google-research/big_vision.git rosbag2dataset/owl/big_vision

4. Download SAM checkpoint

Download sam_vit_b_01ec64.pth:

https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Place it in:

rosbag2dataset/sam/sam_vit_b_01ec64.pth

Data Processing
1. Prepare ROS bag

Bag should include:

RGB-D images

TF

End-effector pose

2. Convert ROS bag to dataset
cd ~/scenerep
python rosbag2dataset/rosbag2dataset_5hz.py [dataname.bag]

3. Run OWL-ViT object scoring
python rosbag2dataset/owl/owl_object_score.py [dataname]

4. Run SAM segmentation
python rosbag2dataset/sam/sam.py [dataname]

Demo

Edit a config file under:

config/[dataname.config]


Run:

python data_demo.py --config config/[dataname.config]
