# Multi-Modal Molecular Property Prediction

This project implements a multi-modal machine learning pipeline designed to predict the properties of small molecules (specifically **BACE-1 inhibitors**) by integrating structural chemical data with AI-generated textual descriptions. 

By leveraging Large Language Models (LLMs) and specialized chemical transformers, this framework captures both the explicit physical characteristics and the implicit semantic context of molecular structures.

---

## 🏗 Workflow Architecture

The pipeline is divided into three primary functional blocks: **Data Preparation**, **Feature Generation**, and **Modeling & Analysis**.

### 1. Data Preparation
* **Data Acquisition**: Download and curation of BACE-1 inhibitor datasets.
* **RDKit Fact Extraction**: Preprocessing SMILES strings to extract fundamental chemical properties (Molecular Weight, LogP, Formula, and specific functional group counts).

### 2. Feature Generation (The Multi-Modal Core)
This project utilizes a dual-track embedding strategy:
* **Textual Track**: 
    * **Llama-3.1-8B-Instruct**: Generates professional, natural language descriptions of energetic and chemical characteristics based on RDKit facts.
    * **BGE-M3**: Transforms the LLM-generated text into high-dimensional semantic embeddings (1024-dim).
* **Structural Track**:
    * **ChemBERTa-2 (100M-MLM)**: Extracts hidden-layer representations directly from molecular structural data to capture chemical intuition.

### 3. Modeling and Analysis
* **Individual Modality Regression**: 5-fold cross-validation performed separately on textual embeddings and structural embeddings to establish baselines.
* **Multi-Modal Fusion**: Concatenation of textual and structural vectors to create a fused embedding space.
* **Fused Regression**: Final predictive modeling using combined modalities to achieve superior accuracy in pIC50 prediction.

---

## 🛠 Tech Stack

* **Cheminformatics**: [RDKit](https://www.rdkit.org/)
* **LLM Inference**: [Ollama](https://ollama.com/) (Running Llama-3.1-8B)
* **Embeddings**: [BGE-M3](https://huggingface.co/BAAI/bge-m3) via `sentence-transformers`, [ChemBERTa-2](https://huggingface.co/deepchem/ChemBERTa-77M-MLM)
* **Data Science**: Pandas, Scikit-learn (Regression, 5-fold CV, PCA)
* **Infrastructure**: Local development with scaling to **HPC (High-Performance Computing)** for large-scale inference.

---

## 🚀 Key Features

* **Resilient Pipeline**: Includes checkpointing logic (`processed_indices.txt`) to allow batch processing of 1,500+ molecules to resume seamlessly after interruptions.
* **HPC Optimized**: Scripting logic designed to migrate from local environments to HPC clusters for intensive LLM workloads.
* **Contextual Grounding**: LLM prompts are strictly grounded in calculated RDKit facts to mitigate hallucination and ensure chemical accuracy.

---

## 📊 Results Summary

The project benchmarks the predictive power of different data modalities:
| Modality | Model | Goal |
| :--- | :--- | :--- |
| **Textual Only** | BGE-M3 + Regression | Capture semantic context |
| **Structural Only** | ChemBERTa + Regression | Capture atomic patterns |
| **Multi-Modal** | **Fused Embeddings** | **Optimized pIC50 Prediction** |

---

## 📂 Getting Started

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/molecular-property-prediction.git](https://github.com/your-username/molecular-property-prediction.git)
    ```
2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Ollama**
    Ensure Ollama is running locally with the Llama-3.1 model:
    ```bash
    ollama run aiasistentworld/Llama-3.1-8B-Instruct-STO-Master
    ```
4.  **Execute Pipeline**
    ```bash
    python main_pipeline.py
    ```

---

## 📜 References
* [cite_start]*Multi-Modal Learning for Predicting Molecular Properties: Integrating Chemical, Visual, and Textual Data* (Coil et al., 2026). [cite: 1]
* [cite_start]*Mol2Vec: Unsupervised Machine Learning Approach with Chemical Intuition*. [cite: 53]
* *ChemBERTa-2: Towards Safe and Effective Molecular Property Prediction*.
