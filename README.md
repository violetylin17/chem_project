# Multi-Modal Molecular Property Prediction

This project implements a multi-modal machine learning pipeline designed to predict the properties of small molecules (specifically **BACE-1 inhibitors**) by integrating structural chemical data with AI-generated textual descriptions. 

By leveraging Large Language Models (LLMs) and specialized chemical transformers, this framework captures both the explicit physical characteristics and the implicit semantic context of molecular structures.

---

## 🏗 Workflow Architecture

The pipeline is divided into three primary functional blocks: **Data Preparation**, **Feature Generation**, and **Modeling & Analysis**.

### 1. Data Preparation
* **Data Acquisition**: Download and curation of BACE-1 inhibitor datasets.
* [cite_start]**RDKit Fact Extraction**: Preprocessing SMILES strings to extract fundamental chemical properties including Molecular Weight, LogP, Formula, and specific functional group counts[cite: 124, 139].

### 2. Feature Generation (The Multi-Modal Core)
This project utilizes a dual-track embedding strategy:
* **Textual Track**: 
    * [cite_start]**Llama-3.1-8B-Instruct-STO**: Generates professional, natural language descriptions of molecular and chemical characteristics based on RDKit-calculated facts[cite: 56, 163].
    * [cite_start]**BGE-M3**: Transforms the LLM-generated text into high-dimensional semantic embeddings (1024-dim)[cite: 54, 165].
* **Structural Track**:
    * **ChemBERTa-3**: Extracts hidden-layer representations directly from molecular structural data to capture chemical intuition and foundation model knowledge.

### 3. Modeling and Analysis
* [cite_start]**Individual Modality Regression**: 5-fold cross-validation performed separately on textual embeddings and structural embeddings to establish baselines[cite: 207, 212].
* [cite_start]**Multi-Modal Fusion**: Concatenation of textual and structural vectors to create a fused embedding space[cite: 62, 77].
* [cite_start]**Fused Regression**: Final predictive modeling using combined modalities to achieve superior accuracy in pIC50 prediction[cite: 11, 256].

---

## 🛠 Tech Stack

* **Package Management**: [uv](https://docs.astral.sh/uv/) for fast, reliable Python environment and dependency management.
* [cite_start]**Cheminformatics**: [RDKit](https://www.rdkit.org/) (Release 2026_03_1)[cite: 56, 124].
* [cite_start]**LLM Inference**: [Ollama](https://ollama.com/) running `Llama-3.1-8B-Instruct-STO-Master`[cite: 56].
* [cite_start]**Embeddings**: [BGE-M3](https://huggingface.co/BAAI/bge-m3) via `sentence-transformers`[cite: 165], and **ChemBERTa-3** foundation models.
* [cite_start]**Data Science**: Pandas, Scikit-learn (Regression, 5-fold CV, PCA)[cite: 61, 185, 212].
* **Infrastructure**: Local development with scaling to **HPC (High-Performance Computing)** for large-scale inference.

---

## 🚀 Key Features

* **Resilient Pipeline**: Includes checkpointing logic (`processed_indices.txt`) to allow batch processing of 1,500+ molecules to resume seamlessly after interruptions.
* **HPC Optimized**: Scripting logic designed to migrate from local environments to HPC clusters for intensive LLM workloads.
* [cite_start]**Contextual Grounding**: LLM prompts are strictly grounded in calculated RDKit facts to mitigate hallucination and ensure chemical accuracy[cite: 57, 303].

---

## 📊 Results Summary

The project benchmarks the predictive power of different data modalities:
| Modality | Model | Goal |
| :--- | :--- | :--- |
| **Textual Only** | BGE-M3 + Regression | [cite_start]Capture semantic context [cite: 148, 271] |
| **Structural Only** | ChemBERTa + Regression | [cite_start]Capture atomic patterns [cite: 268] |
| **Multi-Modal** | **Fused Embeddings** | [cite_start]**Optimized pIC50 Prediction** [cite: 40, 302] |

---

## 📂 Getting Started

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/molecular-property-prediction.git](https://github.com/your-username/molecular-property-prediction.git)
    ```
2.  **Install Dependencies with uv**
    ```bash
    uv sync
    ```
3.  **Run Ollama**
    Ensure Ollama is running locally with the master model:
    ```bash
    ollama run aiasistentworld/Llama-3.1-8B-Instruct-STO-Master
    ```
4.  **Execute Pipeline**
    ```bash
    uv run main_pipeline.py
    ```

---

## 📜 References

```bibtex
@software{rdkit_2026_03_1,
  author = {Greg Landrum and others},
  title = {rdkit/rdkit: 2026_03_1 (Q1 2026) Release},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19250388}
}

@misc{alexh2026llama31sto,
  author = {AlexH},
  title = {Llama-3.1-8B-Instruct-STO-Master: Pushing the limits of 8B architectures},
  year = {2026},
  publisher = {HuggingFace}
}

@misc{bge_m3,
  title={BGE M3-Embedding}, 
  author={Jianlv Chen and others},
  year={2024},
  eprint={2402.03216},
  archivePrefix={arXiv}
}

@article{singh2026chemberta3,
  title={ChemBERTa-3: an open source training framework for chemical foundation models},
  author={Singh, Riya and others},
  journal={Digital Discovery},
  year={2026},
  doi={10.1039/D5DD00348B}
}
