# PB-Mamba

PB-Mamba is a novel **locally-globally collaborative state space model** designed for perinatal brain ultrasound image classification. By integrating cluster-aware attention, the Mamba architecture, and multi-axis convolutions, this model effectively addresses the challenges of speckle noise, semantic poverty, and blurred tissue boundaries in ultrasound imaging.

## Features
- **Local-Global Collaboration**: Synergistically fuses global features (via Mamba) and local features (via Multi-axis CNN) to capture both long-range dependencies and fine-grained boundary details.
- **Cluster-Aware Spatial Attention (CASA)**: Refines input tokens using cluster-adaptive convolutions to enhance semantic quality before processing.
- **Geometric Wave Scanning (GWS)**: A novel scanning mechanism designed to preserve the spatial adjacency and continuity of 2D ultrasound images, overcoming the limitations of standard 1D scans.
- **Robust Performance**: Significantly outperforms baseline models (e.g., MedViT, Swin-Transformer) across three challenging ultrasound datasets.
- **Efficient Architecture**: Leverages the linear complexity of State Space Models (SSMs) for efficient long-sequence modeling.

## About

Perinatal brain ultrasound imaging holds significant value in early screening for neurodevelopmental abnormalities. However, automated classification remains challenging due to inherent speckle noise and blurred tissue boundaries. Traditional State Space Models (SSMs), while efficient, often disrupt spatial continuity and neglect local texture details when applied to 2D images.

**PB-Mamba** addresses these limitations through a holistic design:
1.  **Semantic Refinement**: A Cluster-Aware Spatial Attention layer refines input tokens to mitigate semantic poverty.
2.  **Collaborative Modeling**: The core unit features a **Global Path** (using Mamba with Geometric Wave Scanning) to model anatomical layout, and a **Local Path** (using multi-axis convolutions) to capture critical edge and texture information.

This approach ensures that the model "sees" both the overall anatomical structure and the subtle diagnostic indicators, making it highly effective for complex medical image analysis.

## Results

Extensive experiments on three datasets (Anterior Horn, Posterior Horn, and Fetal Brain Biometry) demonstrate state-of-the-art performance.

*Key Performance Highlights (Fetal Brain Biometry Dataset):*
- **Overall Accuracy (OA)**: 88.2% - **F1-Score**: 86.9% - **AUC of ROC**: 0.961
- **Precision**: 87.2%

*Performance visualization (ROC & PRC Curves):*
![Performance Curves](path/to/your/image.png) ## Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/YukiNeon/PBMamba.git](https://github.com/YukiNeon/PBMamba.git)
   cd PBMamba
