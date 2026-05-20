# Industry Project - Jingwen Wang

# Green Screen Spill Suppression & Video Matting

A deep learning project exploring green screen spill suppression and video matting,
implemented across three model versions with progressively refined approaches.

---

## Model Versions

| Version | Module | Description |
|---------|--------|-------------|
| v1 | SpillNet | CNN-based spill suppression model trained on synthetic green screen data |
| v2 | SAM2-based | Alpha matte generation leveraging Segment Anything Model 2 |
| v3 | MattingNet | Refined matting network with improved alpha prediction and edge detail |

---

## Repository Structure

├── stage1_spillnet/
│   ├── model_spillnet.py
│   ├── loss_spillnet.py
│   ├── train_spillnet.py
│   ├── eval_spillnet.ipynb
│   └── eval_spillnet_video.ipynb
├── stage2_sam2/
│   ├── generate_alpha_sam2.py
│   └── eval_sam2_pipeline.ipynb
├── stage3_mattingnet/
│   ├── model_mattingnet.py
│   ├── loss_mattingnet.py
│   ├── train_mattingnet.py
│   ├── infer_mattingnet.py
│   └── eval_mattingnet.ipynb
├── dataset.py
├── postprocess.py
└── requirements.txt


## Requirements

```bash
pip install -r requirements.txt

## Requirements
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .


