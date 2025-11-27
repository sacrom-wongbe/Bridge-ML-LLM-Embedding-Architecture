# **Bridge-ML-LLM-Embedding-Architecture**

A hybrid machine learning + NLP architecture for predicting seismic vulnerability in U.S. bridges by merging **National Bridge Inventory (NBI)** structural data, **engineer-labeled vulnerability annotations**, and **transformer-based text embeddings** of bridge descriptions.
Developed as part of ongoing research with the **University of Washington (UW) Industrial & Systems Engineering Department**.

---

## **📌 Overview**

Bridge inspection datasets often include both **numerical structural attributes** and **unstructured text fields** describing materials, conditions, damage, and special features. Traditional ML models struggle to integrate these modalities effectively, especially when predicting complex outcomes like **seismic resilience**.

This project builds a unified modeling architecture that:

* Extracts transformer-based embeddings from textual NBI fields
* Engineers domain-specific features based on civil/structural engineering input
* Fuses text embeddings with numerical NBI metrics
* Produces ML-ready datasets for seismic vulnerability classification
* Evaluates multiple model families (tabular, deep, and hybrid approaches)

The goal is to advance seismic risk modeling by **augmenting engineering datasets with LLM-derived semantic structure**, improving predictive power without requiring expensive manual labeling.

---

## **🎯 Research Objectives**

* Recover meaningful semantic information from unstructured NBI text (e.g., material descriptions, structural notes, condition remarks)
* Build transformer-based embedding pipelines compatible with large tabular datasets
* Develop automated feature engineering workflows for structural metrics
* Fuse numerical + embedding spaces to build hybrid ML models
* Assess improvements in seismic vulnerability prediction accuracy compared to numerical-only baselines
* Demonstrate the feasibility of using LLM embeddings in civil infrastructure modeling

---

## **🧠 Methodology & Architecture**

### **1. Text Field Preprocessing**

* Normalize raw NBI text fields
* Remove boilerplate or code-style entries
* Identify fields with high information density for embedding
* Map engineer-labeled bridge categories to unified formats

### **2. Transformer-Based Embeddings**

Embeddings generated from:

* **Sentence-BERT / MPNet**
* **MiniLM**
* **RoBERTa variants**

### **3. Numerical Feature Engineering**

Feature sets include:

* Structural characteristics (span length, deck width, construction year)
* Material indicators
* Condition ratings (superstructure, substructure, deck)
* Seismic design categories + zone mappings
* Age, rehabilitation history, traffic load

### **4. Embedding + Tabular Fusion**

Fused dataset combines:

```
[ numerical engineering features ] + [ transformer embeddings ]
```

Evaluated fusion strategies:

* Direct concatenation
* Dimensionality reduction before fusion
* Weighted attention between modalities

### **5. ML Modeling**

Models tested:

* XGBoost
* Random Forest
* Logistic Regression Baselines
* Feed-forward neural networks on fused embeddings
* Hybrid models trained on mixed-modality vectors

Metrics:

* F1 score on seismic vulnerability classes
* ROC-AUC
* Calibration & stability across train splits

---

## **🧪 Key Findings (So Far)**

* Transformer embeddings capture **structural semantics** not present in numerical fields
* Fused models consistently outperform numerical-only baselines
* PCA-reduced embeddings (20–50 components) yield best performance-to-complexity ratio
* Some NBI text fields are highly predictive after embedding (e.g., “structure notes”, “design description”)
* Embeddings help reduce label scarcity by bringing unlabeled bridges closer to labeled ones in vector space

---

## **🔬 Research Context**

Conducted under the **UW Disaster Data Science Lab / Industrial & Systems Engineering**.
Supports ongoing work in:

* Infrastructure risk modeling
* Natural hazards resilience
* Critical infrastructure ML

---

## **🛠️ Tools & Technologies**

* **Python, pandas, NumPy**
* **sentence-transformers, Hugging Face Transformers**
* **scikit-learn, XGBoost**
* **UMAP, PCA, FAISS (optional for ANN search)**
* **Jupyter Lab**

---

## **📈 Future Work**

* Try larger embedding models (e.g., Instructor-XL, E5-Large)
* Graph-based modeling (GNNs on bridge connectivity networks)
* Incorporate hazard exposure maps (soil type, PGA values)
* Benchmark against expert-engineered seismic models
* Prepare preprint + potential conference poster

---
