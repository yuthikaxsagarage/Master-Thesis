# Guiding Visual Odometry through Classical Methods and Attention

A deep learning approach to visual odometry that combines neural networks with attention layer and classical computer vision techniques for improved camera pose estimation and depth perception.

[![Demo Video](https://i.ytimg.com/vi/lyk9tW-3_QM/maxresdefault.jpg)](https://www.youtube.com/watch?v=lyk9tW-3_QM "Visual Odometry Demo")

## Project Overview

This project implements a visual odometry system that estimates camera motion and depth from image sequences. It leverages self-supervised learning methods to improve upon classical visual odometry techniques. The approach combines:

1. Deep neural networks for depth estimation
2. Pose estimation with attention mechanisms
3. Classical feature-based visual odometry
4. Self-supervised learning using photometric consistency

## Key Features

- Monocular depth estimation using ResNet-based architectures
- Camera pose estimation with attention mechanisms
- Integration with classical feature-based visual odometry
- Self-supervised training without ground truth depth
- Compatible with KITTI and other datasets
- GUI tools for visualization and evaluation

## Technical Details

### Attention Network for Pose Estimation

This project introduces a novel pose attention network architecture:

- **Attention Mechanism**: The network enhances standard CNN-based pose regression with a self-attention mechanism that focuses on the most relevant spatial features for improved pose estimation.
  
- **Architecture Structure**:
  - ResNet encoder (18 or 50 layers) extracts features from image pairs
  - PoseCNN decoder processes features to estimate 6-DoF pose
  - Self-attention layer transforms features through query, key, and value projections
  - Attention matrix created through matrix multiplication and softmax normalization
  - Output combined with original features and refined through convolutional layers

- **Combined Output**: The final pose is a weighted combination of standard CNN pose estimation and attention-enhanced pose, creating more robust estimates.

### Rotation Consistency with Classical VO

The system uniquely integrates learning-based methods with classical visual odometry:

- **Classical VO Integration**:
  - Uses SIFT feature detection and Lucas-Kanade optical flow for feature tracking
  - Essential matrix computation with RANSAC for robust estimation
  - Recovers rotation matrices using OpenCV's `recoverPose` function

- **Rotation Consistency Loss**:
  - Compares rotation matrices from neural network with those from classical VO
  - Measures angular difference between rotation matrices in degrees
  - Works for both forward and backward directions
  - Acts as a regularizer during training

- **Benefits**:
  - Neural network learns rotations consistent with geometry-based methods
  - Provides self-supervision without requiring ground truth pose labels
  - Creates a hybrid system where classical computer vision guides the learning process
  - Significantly improves accuracy for rotation components which are challenging for pure learning-based methods

## Project Structure

- `train.py`: Main training script
- `classic_vo.py`: Classical visual odometry implementation
- `models/`: Neural network architectures
  - `DispResNet.py`: Disparity/depth estimation network
  - `PoseAttentionNet.py`: Pose estimation network with attention mechanism
- `datasets/`: Data loading utilities
- `loss_functions.py`: Self-supervised loss implementations including rotation consistency loss
- `inverse_warp.py`: Differentiable warping for view synthesis
- `gui.py` & `vo_gui.py`: Visualization tools
- `kitti_eval/`: Evaluation tools for the KITTI dataset

## Requirements

- Python 3.6+
- PyTorch 1.0+
- OpenCV
- NumPy
- TensorboardX
- CUDA enabled GPU (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/visual-odometry.git
cd visual-odometry

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py /path/to/dataset --name experiment_name --epochs 200 --batch-size 4 --rotation-consistency-weight 0.2
```

The `rotation-consistency-weight` parameter controls how much the classical VO guidance influences the learning process.

### Evaluation

```bash
python test_vo.py --pretrained-disp /path/to/dispnet/checkpoint --pretrained-pose /path/to/posenet/checkpoint
```

### Visualization

```bash
python vo_gui.py --pretrained-disp /path/to/dispnet/checkpoint --pretrained-pose /path/to/posenet/checkpoint
```

## Dataset Preparation

The code supports KITTI dataset format. To prepare your data:

1. Download the [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)
2. Organize the data according to the expected folder structure
3. Use the provided data loading utilities for custom datasets

## Citation

If you use this code in your research, please cite:

```
@article{visual_odometry,
  title={Guiding Visual Odometry through Classical Methods and Attention},
  author={yuthika Sagaarge},
  journal={engrXiv preprint},
  year={2022}
}
```

## License

MIT

## Acknowledgments

- The code builds upon several open-source projects in the field of self-supervised depth estimation and visual odometry.
- Parts of the implementation are inspired by [Monodepth2](https://github.com/nianticlabs/monodepth2) and other related works.

