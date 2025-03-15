# difflearn

Federated Learning System for Electronic Health Records (EHR)

## Project Vision

The project aims to develop a Federated Learning (FL) system integrated with Differential Privacy (DP) to enhance privacy while ensuring robust performance and scalability across diverse datasets. Key features include:

  - **Privacy Preservation:**

    Employing differential privacy mechanisms to protect sensitive health data.
  - **Handling Non-IID Data:**

    Utilizing demographic-aware gradient clustering to manage data heterogeneity across different institutions.
  - **Performance Maintenance:**

    Implementing dynamic noise adjustment to optimize the balance between privacy and model accuracy.

## Problem Statement

The rise of Electronic Health Records (EHRs) has transformed healthcare by providing rich data for AI applications. However, integrating EHR data across institutions is challenging due to:

  - **Privacy Risks:**

    Even without centralizing data, model updates can inadvertently expose sensitive information.
  - **Non-IID Data:**

    Datasets vary significantly across institutions, influenced by demographic and procedural differences.
  - **Scalability Constraints:**

    Implementing FL with DP introduces computational overhead, particularly challenging for resource-constrained settings.

## Project Scope

The project is structured into three phases over a three-month period, each incrementally enhancing system functionality:

  - **Core Implementation & Baseline Validation:**

    Establishing basic FL and DP frameworks and validating initial performance.
  - **Non-IID Data Optimization & Visual Analytics:**

    Developing clustering strategies to manage data skew and creating visualization tools for model insights.
  - **Integrated Prototyping & Paper Drafting:**

    Assembling all components into a cohesive system, validating with real-world data, and drafting an academic publication to document findings.

## Features in Detail

  - **Differential Privacy Engine:**

    Implements adaptive noise scaling and privacy auditing to protect data.
  - **Gradient Clustering Module:**

    Groups similar client updates to enhance model aggregation efficiency.
  - **Visualization Pipeline:**

    Provides dynamic dashboards and static visualizations for monitoring model performance and resource usage.

## Installation

1) Environment Setup

    ```bash
    conda create -n ehr_fl python=3.10 -y
    conda activate ehr_fl
    conda install -c apple tensorflow-deps
    pip install -r requirements.txt
    ```

2) Run Application

    ```bash
    streamlit run streamlit_app.py
    ```

3) Clean Cache (in between runs)

    ```bash
    find . -type d -name "__pycache__" -exec rm -r {} +
    pip cache purge
    rm -rf logs plots
    ```

## Usage

**Data Loading and Preprocessing**

  - Script: data_loader.py
  
  Loads EHR datasets, handles missing values, encodes categorical variables, scales features, and partitions data into non-IID client subsets.

**Model Training**

  - Function: simulate_client_training()
  
  Simulates distributed training across multiple clients, computes differentially private gradients using a custom DPAdam optimizer, and aggregates model updates.

**Visualization and Monitoring**

  - Utilize Matplotlib, Seaborn, and Plotly to:
      - Track model performance metrics (accuracy, AUC-ROC).
      - Monitor privacy budget consumption (ε).
      - Visualize resource usage (CPU, memory) in real time.

**Validation & Testing**
  - Integrated using TensorFlow Federated and TensorFlow Privacy.
  - Performed comparative analysis using AUC-ROC curves to validate model performance.

**Future Work**
  - Extend to clinical settings for further validation.
  - Explore advanced clustering and personalization techniques for extreme data skew management.
