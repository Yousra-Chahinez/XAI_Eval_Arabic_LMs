# Exploring Explainability in Arabic Language Models: An Empirical Analysis of Techniques

This repository contains the code and resources for our paper titled "**Exploring Explainability in Arabic Language Models: An Empirical Analysis of Techniques**", accepted at ACLing 2024.

# Overview

In this paper, we evaluate various explanation methods based on their faithfulness and plausibility in the context of Arabic Language Models. We fine-tune Arabic language models on two specific tasks: Arabic Sentiment Analysis (ASA) and Semantic Question Similarity (Q2Q). For each combination of model and task, we apply several explanation methods and then evaluate them using the following criteria:
- Faithfulness: Measured using the Area Under the Threshold-Performance Curve (AUC-TP).
- Plausibility: Measured using Mean Average Precision (MAP).

# Datasets 
- Hotel Arabic-Reviews Dataset (HARD): Used for the Arabic Sentiment Analysis (ASA) task.
- Mawdoo3 Q2Q Dataset (MQ2Q): Used for the Semantic Question Similarity (Q2Q) task.

# Models
- AraBERT
- AraGPT2

# Explanation Methods
1. Gradient-based methods: Saliency, Input*Gradient, Integrated Gradients
2. Perturbation-based methods: Local Interpretable Model-Agnostic Explanations (LIME). SHapley Value Sampling (SHAP_VS): Estimates Shapley values using random sampling.

# Repository Structure
- data/: Contains the datasets for ASA and Q2Q tasks.
- models/: 
- explanation_methods/: Implementations of different explanation methods.
- evaluation_eval/: Scripts for evaluating faithfulness and plausibility.
- utils/: Scripts for common functions and visualization.


# Arabic NLP Model Training and Evaluation Pipeline

This project provides scripts for training Arabic NLP models and constructing human rationales. Below are examples of how to run the provided scripts from the command line.

## Training Script

To train a model on your dataset, use the following command:
```bash
python main.py --model_name "aubmindlab/bert-base-arabertv2" --dataset_path "data/HARD_balanced-reviews.tsv" --task_type "ASA" --reduce_data --reduce_size 17000
```
To construct human ratioanles from human annotations, use the following command:
The output of this command is on data/ASA_human_ratioanles.csv
```bash
python construct_rationale.py --model_name aubmindlab/bert-base-arabertv2 --csv_path data/annotations.csv --output_path data/processed_rationales.csv
```


# Citation
If you use this repository or our paper in your work, please consider citing:

# Contact
For any inquiries or collaborations, feel free to reach out to us at \email{hadjazzem.yousra@gmail}.com.

