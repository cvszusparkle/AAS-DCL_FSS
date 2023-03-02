# AAS-DCL
The paper: "[**Dual Contrastive Learning with Anatomical Auxiliary Supervision for Few-shot Medical Image Segmentation**](https://link.springer.com/chapter/10.1007/978-3-031-20044-1_24)"  
More contents will be added later.

# Requirements
```
CUDA/CUDNN
torch >= 1.4.0
torchvision
numpy
```

# Datasets
* **CHAOS-T2**: A abdominal MRI dataset from [CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation](https://chaos.grand-challenge.org/)
* **Synapse**: A abdominal CT dataset from [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)

# Acknowledgement
Part of the code about the local prototype is referenced from [SSL-ALPNet](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation).  
The contrastive loss is based on [info-NCE](https://github.com/RElbers/info-nce-pytorch)


# Citation
@inproceedings{wu2022dual,  
  title={Dual Contrastive Learning with Anatomical Auxiliary Supervision for Few-Shot Medical Image Segmentation},  
  author={Wu, Huisi and Xiao, Fangyan and Liang, Chongxin},  
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XX},  
  pages={417--434},  
  year={2022},  
  organization={Springer}  
}  

