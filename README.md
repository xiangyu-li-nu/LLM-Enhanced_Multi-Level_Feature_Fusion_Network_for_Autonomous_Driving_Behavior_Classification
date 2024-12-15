
# LLM-MLFFN: Large Language Model-Enhanced Multi-Level Feature Fusion Network

This repository contains the code and data for the research paper titled **"Large Language Model-Enhanced Multi-Level Feature Fusion Network for Autonomous Driving Behavior Classification"**. The work focuses on leveraging multimodal feature fusion and large language models (LLMs) to classify autonomous vehicle (AV) driving behaviors with high accuracy. The proposed **LLM-MLFFN** framework achieves state-of-the-art results by integrating numerical features, semantic descriptions, and advanced attention mechanisms.

## Overview

Accurate classification of AV driving behaviors is critical for optimizing driving algorithms and enhancing road safety. This repository provides a comprehensive implementation of the **LLM-MLFFN** framework, which consists of:
1. **Multi-Level Feature Extraction**: Extracts numerical features, including statistical, behavioral, and dynamic aspects of driving data.
2. **Semantic Description Module**: Leverages large language models (e.g., GPT-4) to transform numerical features into high-level semantic representations.
3. **Dual-Channel Multi-Modal Feature Fusion**: Combines numerical and semantic features using weighted attention mechanisms for robust classification.

The model is evaluated on the **Waymo Open Trajectory Dataset**, achieving a classification accuracy of **94%**, significantly surpassing existing benchmarks.

---

## Author

This work was conducted by **Xiangyu Li**, a second-year Ph.D. student in Transportation Engineering at Northwestern University. For any inquiries or collaborations, feel free to contact the author:

- **Email**: xiangyuli2027@u.northwestern.edu

---

## Directory Structure

- **`Ablation_experiment/`**  
  Contains scripts for analyzing the contributions of individual modules within the LLM-MLFFN framework.
  - `Complete_model.py`: Implements the full LLM-MLFFN model.
  - `Only_numerical_features.py`: Tests the model using only numerical features.
  - `Only_text_features.py`: Tests the model using only semantic features.
  - `Remove_multi-scale_convolution.py`: Ablation study for removing multi-scale convolutions.
  - `Remove_spatiotemporal_attention.py`: Ablation study for removing spatio-temporal attention mechanisms.
  - `extracted_features_with_analysis.csv`: Preprocessed feature dataset used for experiments.

- **`Behavior_data_set/`**  
  Contains categorized driving behavior data.
  - `Aggressive/`, `Assertive/`, `Conservative/`, `Moderate/`: Subfolders for different driving styles, each containing raw and processed data.

- **`Comparative_experiment/`**  
  Scripts for comparing LLM-MLFFN with baseline models.
  - `FCN.py`, `LSTM.py`, `GRU-FCN.py`, `MLP.py`, etc.: Implementations of baseline models such as Fully Convolutional Networks (FCN), Long Short-Term Memory networks (LSTM), and Multi-Layer Perceptrons (MLP).
  - `extracted_features.csv`: Dataset used for model comparisons.

- **`LLM-MLFFN/`**  
  Core implementation of the proposed framework.
  - `LLM-MLFFN.py`: Main script for training and evaluating the model.
  - `Feature_extraction.py`: Multi-level feature extraction module.
  - `Dual_channel_network.py`: Dual-channel architecture for feature fusion.
  - `GPT_content_analysis.py`: Semantic description module using GPT-based prompts.
  - `Data_processing.py`: Data preprocessing and normalization steps.

- **`Plot/`**  
  Scripts and results for visualizing experimental findings.
  - `ablation_comparison.png`: Visualization of the ablation study results.
  - `Model_comparison_chart.py`: Generates comparison plots between baseline models and LLM-MLFFN.
  - `Feature_analysis.py`: Analyzes feature distributions across driving behavior types.
  - `extracted_features_with_analysis.csv`: Dataset for generating plots.

---

## Dataset

The experiments use the **Waymo Open Trajectory Dataset**, processed and filtered for meaningful driving behavior analysis. Preprocessed data files are included in the repository under `Behavior_data_set/`.

---

