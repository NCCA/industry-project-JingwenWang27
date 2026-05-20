````markdown
# Green Screen Spill Suppression and Video Matting

This repository contains the implementation for an industry project exploring AI-assisted green screen spill suppression and video matting. The project focuses on removing green colour contamination from foreground subjects after chroma keying, especially around hair, semi-transparent edges, and reflective regions.

The project was developed as a staged pipeline. Stage 1 is the main implemented solution, while Stage 2 and Stage 3 are exploratory extensions.

## Project Overview

Green screen compositing often produces residual green spill around foreground boundaries. Traditional keying tools can generate an alpha matte, but they may still leave colour contamination on the foreground. This project investigates whether a small neural network can use the original image and alpha information to predict a cleaner foreground result.

The final practical solution is based on SpillNet, a CNN-based model designed for foreground colour correction rather than full segmentation. The alpha matte is used as guidance so that the network can focus on despill and edge refinement.

## Repository Structure

```text
stage1_spillnet/
  model_spillnet.py
  loss_spillnet.py
  train_spillnet.py
  eval_spillnet.ipynb
  eval_spillnet_video.ipynb

stage2_sam2/
  generate_alpha_sam2.py
  eval_sam2_pipeline.ipynb

stage3_mattingnet/
  model_mattingnet.py
  loss_mattingnet.py
  train_mattingnet.py
  infer_mattingnet.py
  eval_mattingnet.ipynb

dataset/
  test/

dataset.py
postprocess.py
requirements.txt
````

## Model Versions

| Stage   | Module        | Role                                                            |
| ------- | ------------- | --------------------------------------------------------------- |
| Stage 1 | SpillNet      | Main supervised green spill suppression model                   |
| Stage 2 | SAM2 pipeline | Exploratory automatic alpha matte generation                    |
| Stage 3 | MattingNet    | Exploratory dual-output matting and foreground prediction model |

## Installation

Create a Python environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

If running the SAM2-based experiment, install SAM2 separately:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

## Dataset

The full training dataset is not included in this repository due to size constraints. A small set of sample test images is included under:

```text
dataset/test/
```

The expected dataset structure for training is:

```text
dataset/
  image/
  fg/
  alpha/
  test/
```

Where:

* `image/` contains green screen input images
* `fg/` contains clean foreground target images
* `alpha/` contains alpha matte images
* `test/` contains sample images for inference

## Training

To train the main SpillNet model:

```bash
python stage1_spillnet/train_spillnet.py
```

To train the exploratory MattingNet model:

```bash
python stage3_mattingnet/train_mattingnet.py
```

Model checkpoints are saved under:

```text
checkpoints/
```

Checkpoints are not included in this repository because of file size limits.

## Inference

To run inference with the exploratory MattingNet model:

```bash
python stage3_mattingnet/infer_mattingnet.py
```

The script reads sample images from:

```text
dataset/test/
```

and writes output images to:

```text
dataset/test_output_joint/
```

## Evaluation Notebooks

The repository includes notebooks for visual inspection and comparison:

```text
stage1_spillnet/eval_spillnet.ipynb
stage1_spillnet/eval_spillnet_video.ipynb
stage2_sam2/eval_sam2_pipeline.ipynb
stage3_mattingnet/eval_mattingnet.ipynb
```

These notebooks are used to compare input images, alpha mattes, model outputs, and target foreground images.

## Limitations

The current project is a prototype rather than a production-ready keying tool. The model depends on the quality of the provided alpha matte, and the training dataset is limited in scale and diversity. Difficult cases such as fine hair, motion blur, transparent objects, and strong green reflections remain challenging.

The SAM2 and MattingNet stages are exploratory experiments. They demonstrate possible future directions, but the main working solution of the project is Stage 1 SpillNet.

## References

Third-party libraries and models used in this project include PyTorch, OpenCV, Pillow, Matplotlib, and Segment Anything Model 2 for the exploratory alpha generation stage.

```
```
