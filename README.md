# FEAT: Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions
This is the implementation of the approach described in the paper "Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions" by Ye, Han-Jia et al. The proposed approach adapts instance embeddings to the target classification task with a set-to-set function and achieves state-of-the-art results on multiple few-shot learning benchmarks.

## Installation
Required packages:
* Pytorch
* tensorboardX
* Numpy
 
## Dataset
Once you have downloaded the dataset, you will need to create a new folder named "images" within the "minimagenet" or "retail" folder, and place all of the images into this folder. The data loader that we have provided will automatically read the images from the "images" folder.


The results on the MiniImageNet and TieredImageNet datasets are shown below:

## MiniImageNet
| Model | 1-Shot 5-Way	| 5-Shot 5-Way |
|:------:|:-------------:|:------------:|
| ProtoNet |	62.21 |	80.64 |
| BILSTM |	63.04 |	80.63 |
| DEEPSETS |	64.24 |	80.51 |
| GCN |	63.93 |	81.65 |
| FEAT |	66.08 |	81.95 |

## TieredImageNet
| Model | 1-Shot 5-Way	| 5-Shot 5-Way |
|:------:|:-------------:|:------------:|
| ProtoNet |	67.93 |	84.23 |
| BILSTM |	67.84 |	83.53 |
| DEEPSETS |	68.89 |	84.86 |
| GCN	| 66.20 |	84.64 |
| FEAT |	70.23 |	84.37 |

### References
   Ye, Han-Jia, et al. "Few-shot learning via embedding adaptation with set-to-set functions." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
