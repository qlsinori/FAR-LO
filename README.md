# FAR-LO
Neural LiDAR Odometry with Feature Association and Reuse for Unstructured environments. Coming soon !!!!!!!
<img width="1268" alt="overall3" src="https://github.com/user-attachments/assets/29728bc5-d1df-4da3-9f94-b9c1e37ac9c7" />

## Abstract
Odometry plays a crucial role in autonomous tasks of field robots, providing accurate position and orientation derived from sequential sensor observations. Odometry based on Light Detection and Ranging (LiDAR) sensors has demonstrated widespread applicability in environments with rich structured features, such as urban and indoor settings. However, for unstructured environments like scrubland and rural roads, the extraction, description, and correct matching of LiDAR features between frames become challenging. Due to the lack of flat surface and straight lines, the existing odometry approaches, whether using hand-crafted features such as edge and planar points or learned features through networks, will face the problem of decreased positioning accuracy and potential failure. Therefore, we propose a neural LiDAR odometry based on Trans-frame Association to extract more effective features for pose estimation in unstructured environments. The Trans-frame Association module contains a fully interactive frame Transformer and a scan‐aware Swin Transformer. The former applies cross-attention to features extracted from two consecutive frames, thus enhancing the accuracy and robustness of feature correspondences by considering the contextual information. The latter restricts the attention mechanism to shift along the scan lines of LiDAR, thereby leveraging the sensor's inherent higher horizontal resolution. Our Transformer has linear complexity, which guarantees the module can meet real-time requirements. Additionally, we design a Reuse Refinement Pyramid architecture to further improve the accuracy of pose estimation by reusing multi-resolution features. We conducted extensive experiments on the RELLIS‐3D dataset and our Matian Ridge dataset collected in a representative unstructured scene. The results demonstrate that our network outperforms recent learning‐based LiDAR odometry methods in terms of accuracy.

## Installation
Our model only depends on the following commonly used packages.

| Package      | Version                          |
| ------------ | -------------------------------- |
| CUDA         |  12.2                            |
| Python       |  3.7.13                          |
| PyTorch      |  1.12.0                          |
| h5py         | *not specified*                  |
| tqdm         | *not specified*                  |
| numpy        | *not specified*                  |
| openpyxl     | *not specified*                  |

Device: Tesla V100-PCIE-32GB*8

## Install the pointnet2 library
Compile the required library [Reference: [RegFormer] (https://github.com/IRMVLab/RegFormer)].
```bash
cd pointnet2
python setup.py install
cd ops_pytorch
cd fused_conv_random_k
python setup.py install
cd ../
cd fused_conv_select_k
python setup.py install
```


## Datasets
### RELLIS-3D Dataset
Datasets are available at RELLIS-3D Dataset website: (https://github.com/unmannedlab/RELLIS-3D) The data of the RELLIS-3D Dataset should be organized as follows:

```
data_root
├── 00
│   ├── os1_cloud_node_kitti_bin
│   ├── calib.txt
│   ├── poses.txt
├── 01
├── ...
```


## Training
Train the network by running :
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 8 ddp_train_raft.py
```
Please reminder to specify the parameter in the scripts.

## Testing
You can set the parameter 'eval_before' as 'True' in file ddp_configs_raft.py, then evaluate the network by running :
```bash
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node 1 ddp_train_raft.py
```
Please reminder to specify the parameter in the scripts.

## TODO List and ETA
- [x] Inference code and pretrained models (2024-12-13)
- [ ] Code for reproducing the test-set results (2025-3-7)
- [ ] Training code and training data preparation (expected 2025-4-10)


### Acknowledgments
We thank the following open-source project for the help of the implementations:
- [LoFTR](https://github.com/zju3dv/LoFTR)
- [RegFormer](https://github.com/IRMVLab/RegFormer)
- [PointNet++](https://github.com/charlesq34/pointnet2) 
- [KITTI_odometry_evaluation_tool](https://github.com/LeoQLi/KITTI_odometry_evaluation_tool) 
- [PWCLONet](https://github.com/IRMVLab/PWCLONet)
- [HRegNet](https://github.com/ispc-lab/HRegNet)
