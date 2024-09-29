# Exploring Explainability in Arabic Language Models: An Empirical Analysis of Techniques

This repository contains the code and resources for the paper "**Exploring Explainability in Arabic Language Models: An Empirical Analysis of Techniques**," accepted at ACLing 2024.

## Overview

In this study, we evaluate various explanation methods based on their **faithfulness** and **plausibility** within the context of Arabic Language Models (LMs). We fine-tune Arabic LMs on two specific tasks: **Arabic Sentiment Analysis (ASA)** and **Semantic Question Similarity (Q2Q)**. For each model-task combination, we apply several explanation methods and evaluate them using:

- **Faithfulness**: Measured via the Area Under the Threshold-Performance Curve (AUC-TP).
- **Plausibility**: Measured using Mean Average Precision (MAP).

This repository provides a comprehensive pipeline for fine-tuning Arabic LMs, generating explanations, and evaluating them in Arabic NLP contexts. It supports various explanation methods, including Vanilla Gradient, Gradient Input, Integrated Gradients, LIME, and SHAP.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Yousra-Chahinez/XAI_Eval_Arabic_LMs.git
cd XAI_Eval_Arabic_LMs
pip install -r requirements.txt
```

## Repository Structure
XAI_Eval_Arabic_LMs/
│
├── data/                  
│   ├── ASA_human_annotations/ 
│   └── <other datasets>      
│
├── src/
│   ├── components/           
│   │   ├── data_loader.py
│   │   ├── data_processor.py
│   │   ├── model_loader.py
│   │   └── model_trainer.py
│   │
│   ├── explanation_methods/   
│   │   ├── perturbation_methods.py
│   │   └── gradient_methods.py
│   │
│   └── explanation_eval/      
│       ├── faithfulness.py
│       └── plausibility.py
│
├── utils/                  
│
└── main.py                   

## Training the model
To train the model and evaluate its performance, use the following command:
```bash
python main.py train --model_name "aubmindlab/bert-base-arabertv2" --dataset_path "data/HARD_balanced-reviews.tsv" --task_type ASA --seed 42
```

## Generating Explanations
To generate explanations for a specific instance in the test dataset after training, use the command below:

```bash
python main.py explain --model_name "aubmindlab/bert-base-arabertv2" --dataset_path "data/HARD_balanced-reviews.tsv" --task_type ASA --explanation_method vanilla_grad --instance_index 0 --target_class 1
```

## Citation

If you use this repository or our paper in your work, please consider citing us as follows:

```bibtex
@inproceedings{your_citation_key,
  title={Exploring Explainability in Arabic Language Models: An Empirical Analysis of Techniques},
  author={Your Name},
  booktitle={ACLing 2024},
  year={2024}
}
```

## Contact
For inquiries or collaboration opportunities, please reach out via email at [hadjazzem.yousra@gmail.com](mailto:hadjazzem.yousra@gmail.com).
