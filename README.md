# Low-resource finetuning of foundation models beats state-of-the-art in histopathology

This is the repository of  [Low-resource finetuning of foundation models beats state-of-the-art in histopathology](https://arxiv.org/abs/2401.04720) which was accepted at ISBI 2024.
It is a slightly adapted version of the original [DINOv2](https://arxiv.org/abs/2304.07193), GitHub [repository](https://github.com/facebookresearch/dinov2/tree/main/dinov2).
## Finetuning can be compute efficient
<img src="media/figure1.png" alt="Title" title="Finetuning works well" width="500" /> 
We propose finetuning a DINOv2 ViT-S, which yields at least equal performance compared to CTransPath and RetCCL but in a fraction of domain specific training time. Performance is measured on three datasets: TCGA & CPTAC (WSI-level classification) and NCT-CRC (patch-level classification).

## Loss and performance over time
![](media/loss_curves.png "Title")

Performance over time of finetuning a ViT-s with DINOv2: a) on NCT-CRC and evaluating on the external NCT-
CRC testset on patch-level classification and b) on TCGA and testing on TCGA (5-fold cross-validation) and CPTAC (external
testset) on WSI-level classification.


# Model farm
We make all models as well as heads used for training publicly available in the following.

## Pretrained models finetuned on NCT-CRC-100K

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />iterations</th>
      <th>CRC-VAL-HE-7K<br />20-NN balanced acc</th>
      <th>CRC-VAL-HE-7K<br />linear balanced acc</th>
      <th>teacher backbone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">2k</td>
      <td align="right">93.8%</td>
      <td align="right">92.7%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">10k</td>
      <td align="right">93.4%</td>
      <td align="right">93.7%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
  </tbody>
</table>

## Pretrained models finetuned on TCGA

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />iterations</th>
      <th>TCGA<br />AUROC</th>
      <th>CPTAC<br />AUROC</th>
      <th>teacher backbone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">30k</td>
      <td align="right">89%</td>
      <td align="right">85%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">60k</td>
      <td align="right">84%</td>
      <td align="right">79%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
  </tbody>
</table>

## DINO Backbone Loading Function for downstream tasks

The `get_dino_backbone` function is used to load the teacher and student DINO backbone models, adjust positional embeddings, and load pretrained weights into them.
Use the checkpoint.pth files given out from the training as dictonary.

### Function: `get_dino_backbone`

```python
import torch
import torch.nn as nn

def get_dino_backbone(dict_path, device):
    """
    Load the DINO backbone models (teacher and student), correct the state dictionary,
    and adjust the positional embeddings for loading the pretrained weights.

    Args:
        dict_path (str): Path to the dictionary containing the pretrained weights.
        device (str): Device on which to map the model ('cpu' or 'cuda').

    Returns:
        model_teacher (torch.nn.Module): The teacher model loaded with corrected weights.
        model_student (torch.nn.Module): The student model loaded with corrected weights.
    """

    embed_dim = 384  # Embedding dimension for the positional embedding
    
    # Load the pre-trained DINO models for both teacher and student
    model_student = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model_teacher = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    
    # Load the pretrained weights from the provided checkpoint
    pretrained = torch.load(dict_path, map_location=torch.device(device))['model']

    # Extract only the keys related to the teacher model by filtering 'teacher.' prefix
    teacher_state_dict = {k.replace('teacher.', ''): v for k, v in pretrained.items() if k.startswith('teacher.')}
    
    # Debugging: print the keys to verify correct extraction of teacher weights
    print("Keys in teacher state dict:")
    for key in teacher_state_dict.keys():
        print(key)
    
    # Prepare teacher's state dict for loading by removing 'backbone.' prefix
    teacher_state_dict_corrected = {}
    for key, value in teacher_state_dict.items():
        if 'dino_head' in key:
            print('dino_head not used')  # Skipping the classification head
        else:
            new_key = key.replace('backbone.', '')  # Remove 'backbone.' from keys
            teacher_state_dict_corrected[new_key] = value

    # Extract and prepare the student state dictionary in a similar way
    student_state_dict = {k.replace('student.', ''): v for k, v in pretrained.items() if k.startswith('student.')}
    student_state_dict_corrected = {}
    for key, value in student_state_dict.items():
        if 'dino_head' in key:
            print('dino_head not used')  # Skipping the classification head
        else:
            new_key = key.replace('backbone.', '')  # Remove 'backbone.' from keys
            student_state_dict_corrected[new_key] = value

    # Create new positional embeddings with the correct size (1, 257, embed_dim)
    pos_embed1 = nn.Parameter(torch.zeros(1, 257, embed_dim))
    pos_embed2 = nn.Parameter(torch.zeros(1, 257, embed_dim))
    
    # Replace the positional embeddings in the models
    model_student.pos_embed = pos_embed1
    model_teacher.pos_embed = pos_embed2

    # Load the corrected state dictionaries into the models (strict=True to enforce matching keys)
    model_student.load_state_dict(student_state_dict_corrected, strict=True)
    model_teacher.load_state_dict(teacher_state_dict_corrected, strict=True)

    # Return both models; typically the teacher model is used as the backbone
    return model_teacher, model_student
```

## Installation

This requires the same prerequisites as the original DINOv2 implementation.

The training and evaluation code requires PyTorch 2.0 and xFormers 0.0.18 as well as a number of other 3rd party packages. Note that the code has only been tested with the specified versions and also expects a Linux environment. To setup all the required dependencies for training and evaluation, please follow the instructions below:

conda (Recommended) - Clone the repository and then create and activate a dinov2 conda environment using the provided environment definition:

```python
conda env create -f conda.yaml
conda activate dinov2
```
You can also just run the .sh file for cloning the repository and creating the conda enviroment:
[Install Script](run_dinov2)
## Use the pipeline

Currently, the github repository is meant to run on one GPU only. It can simply be run by this line of code once all the hyperparameters are set in the ssl_default_config.yaml.
The path to the folder containing all image patches for the training is given in line 64:

```python
python dinov2/train/train.py --config-file ssl_default_config.yaml --input-dir "PathtoInputdir" --output-dir "PathtoOutputdir"
```

## Continue finetuning

If you want to continue finetuning or use the DINO heads, the remaining weights can be found here:

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th>dataset</th>
      <th># of<br />iterations</th>
      <th>student backbone</th>
      <th>student DINO head</th>
      <th>teacher DINO head</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td>NCT-CRC-100K</td>
      <td align="right">2k</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_student_checkpoint.pth?download=1">student backbone</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_student_dino_head_checkpoint.pth?download=1">student DINO head</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_teacher_dino_head_checkpoint.pth?download=1">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>NCT-CRC-100K</td>
      <td align="right">10k</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_student_checkpoint.pth?download=1">student backbone</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_student_dino_head_checkpoint.pth?download=1">student DINO head</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_teacher_dino_head_checkpoint.pth?download=1">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-S/14</td>
      <td>TCGA</td>
      <td align="right">30k</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_student_checkpoint.pth?download=1">student backbone</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_student_dino_head_checkpoint.pth?download=1">student DINO head</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_teacher_dino_head_checkpoint.pth?download=1">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>TCGA</td>
      <td align="right">60k</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_student_checkpoint.pth?download=1">student backbone</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_student_dino_head_checkpoint.pth?download=1">student DINO head</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_teacher_dino_head_checkpoint.pth?download=1">teacher DINO head</a></td>
    </tr>
  </tbody>
</table>

To load these weights, it is enough to add the path to the config file under head_path. The path that has to be added is to a folder containing the weights. The weights have to be renamed after downloading them for the available code to work (e.g. student_dino_head_checkpoint.pth). More details can be found in the file /dinov2/dinov2/train/ssl_meta_arch.py.

## Citation

If you find our research helpful, please consider citing:

```
@misc{roth2024lowresource,
  title={Low-resource finetuning of foundation models beats state-of-the-art in histopathology},
  author={Benedikt Roth and Valentin Koch and Sophia J. Wagner and Julia A. Schnabel and Carsten Marr and Tingying Peng},
  year={2024},
  eprint={2401.04720},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
# tools_dinov2

