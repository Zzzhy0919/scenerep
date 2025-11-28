# scenerep

## Installation
1. Create environment
```bash
conda create -n scenerep python=3.11
conda activate scenerep
```

2. Install requirements
```bash
cd scenerep
pip install -r requirements.txt
git clone https://github.com/google-research/big_vision.git rosbag2dataset/owl/big_vision
```
Then download the checkpoints [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth). Place it in:rosbag2dataset/sam/sam_vit_b_01ec64.pth

## Data Processing
1. Prepare ROS bag
Bag should include:
RGB-D images
TF
End-effector pose

2. Convert ROS bag to dataset
```bash
cd ~/scenerep
python rosbag2dataset/rosbag2dataset_5hz.py [dataname.bag]
```

4. Run OWL-ViT object scoring & SAM segmentation
```bash
python rosbag2dataset/owl/owl_object_score.py [dataname]
python rosbag2dataset/sam/sam.py [dataname]
```

## Demo

Edit a config file under: config/[dataname.config]

Run:
```bash
python data_demo.py --config config/[dataname.config]
```
