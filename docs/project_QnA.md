# Brain Tumor FL-QPSO Pipeline: Comprehensive Q&A

This document provides detailed explanations of the key concepts, architectures, frameworks, and results from the Brain Tumor FL-QPSO project, addressing all specified topics.

---

## 1. Neural Networks & Core Concepts

### CNN Layers
Convolutional Neural Networks (CNNs) use convolutional layers to extract spatial features from images. In this project, CNNs form the backbone of both the classification module (ResNet-18) and the segmentation module (3D Attention U-Net). They apply learnable filters to input images, creating feature maps that capture everything from simple edges to complex tumor textures.

### T1, T2, T1ce, FLAIR
These are four different Magnetic Resonance Imaging (MRI) modalities used as inputs for the 3D segmentation module:
*   **T1-weighted (T1):** Highlights anatomical structures.
*   **T1-weighted with contrast (T1ce):** Highlights the active, enhancing boundaries of the tumor.
*   **T2-weighted (T2):** Visualizes fluid and edema (swelling) around the tumor.
*   **FLAIR (Fluid Attenuated Inversion Recovery):** Suppresses cerebrospinal fluid to better visualize edema and non-enhancing tumor regions.

### ResNet-18 (11.2M params) & "Per Hospital" Meaning
ResNet-18 is a deep CNN with residual connections (skip connections) that prevent the vanishing gradient problem. 
*   **11.2 Million Parameters:** This refers to the total number of learnable weights and biases in the model architecture.
*   **"Per hospital" meaning:** In the Federated Learning setup, each of the 3 simulated hospitals (clients) runs its own local, isolated copy of this 11.2M parameter model. They train it locally on their private data and only send the updated parameter weights to the central server, ensuring patient data never leaves the hospital.

### 3D Attention U-Net (5.9M params) & Its Focus
A volumetric CNN architecture used for 3D tumor segmentation. 
*   **5.9 Million Parameters:** The number of learnable weights in this specific 3D architecture.
*   **What features it focuses on:** The model integrates **Attention Gates**. These gates learn to automatically suppress irrelevant regions (like healthy brain tissue or background) and amplify/focus solely on the tumor target regions (Whole Tumor, Tumor Core, Enhancing Tumor).

### Long-Range Spatial Dependencies
In medical imaging, a tumor in one part of the brain might deform structures far away, or the context of the whole brain is needed to correctly classify a region. Standard CNNs only look at small local patches. Capturing "long-range spatial dependencies" means the architecture (like Attention U-Net or Attention LSTMs) can connect and correlate information from distant parts of the 3D volume or sequence to make more accurate, context-aware predictions.

---

## 2. Federated Learning & Optimization

### IID vs Non-IID (Heterogeneous) Data
*   **IID (Independent and Identically Distributed):** Ideal data where every hospital has the exact same ratio of tumor types and similar image quality. This is rare in reality.
*   **Non-IID (Heterogeneous Data):** Real-world medical data. Hospital A might have mostly Gliomas, Hospital B might use a different MRI scanner brand, and Hospital C might have 10x more data. This heterogeneity causes standard federated models to bias heavily towards the largest hospital.

### FedAvg (Federated Averaging)
The standard baseline FL aggregation method. The central server computes a dataset-size-weighted average of the weights from all clients. It struggles under Non-IID conditions because the large hospitals dominate the average, leaving smaller hospitals with poor diagnostic accuracy.

### FedProx
An improvement over FedAvg that adds a "proximal regularization term" during local training. It penalizes the client's local model if its weights drift too far from the global model, helping to stabilize training under heterogeneous data conditions.

### QPSO (Quantum Particle Swarm Optimization)
A quantum-inspired swarm optimization algorithm. Instead of just averaging weights, it treats each client's model weights as a "particle" in a quantum potential well. This allows the server to probabilistically explore the solution space and escape local optima using quantum tunneling equations.

### Layer-by-Layer QPSO & Fairness Analysis
Instead of applying QPSO to all 11.2M parameters simultaneously (which acts like random noise), this project's **FedQPSO** algorithm applies it *layer-by-layer* using a combined cross-client validation loss evaluation.
*   **How it works/improves the model:** The server evaluates candidate weight configurations against a combined validation set from *all* clients. Unstable weight combinations are rejected before being committed to the global model.
*   **Fairness Analysis:** Because a weight configuration is rejected if it performs poorly on *any* client's validation data, the algorithm inherently optimizes for the weakest client. In the Natural Heterogeneity setup, FedQPSO reduced the max-min accuracy gap between hospitals by **81%** compared to FedAvg, ensuring smaller hospitals receive equitable diagnostic performance.

---

## 3. Tumor Classification & Progression

### Pituitary, Meningioma, Glioma
The three classes of brain tumors classified by the Federated ResNet-18 module:
*   **Glioma:** Malignant, fast-growing, and highly aggressive (highest clinical urgency).
*   **Meningioma:** Typically benign, slow-growing tumors forming on the membranes covering the brain.
*   **Pituitary:** Tumors in the pituitary gland, often benign but can affect hormone levels and vision.

### HGG vs LGG Models
Used specifically in the longitudinal progression module:
*   **HGG (High-Grade Glioma):** Aggressive, fast-growing malignant tumors requiring rapid intervention.
*   **LGG (Low-Grade Glioma):** Slower-growing, less aggressive tumors with different growth trajectories.

### Math Models: Linear, Logistic, Gompertz, Exponential
Classical mathematical equations used to fit historical tumor volumes and predict future growth trajectories:
*   **Linear:** Assumes a constant absolute growth rate.
*   **Exponential:** Models unrestricted rapid growth (often seen in early-phase aggressive tumors).
*   **Logistic:** Models S-curve growth that slows down as the tumor reaches a maximum carrying capacity due to resource limits.
*   **Gompertz:** Similar to the logistic model but asymmetrical; it is widely used in tumor biology to model growth that slows exponentially.

### Attention LSTM
A Long Short-Term Memory deep learning network enhanced with an Attention mechanism. While math models capture the broad growth trend, the Attention LSTM learns the "residuals"—the complex, non-linear growth patterns and deviations where mathematical models fail. This hybrid approach improves overall prediction accuracy.

---

## 4. Frameworks & Technologies

### MONAI Framework & Sliding Window Inference
**MONAI** is a PyTorch-based open-source framework specifically optimized for healthcare and medical imaging. 
*   **Sliding Window Inference:** High-resolution 3D MRI volumes are too large to fit into GPU memory at once. Sliding window inference breaks the 3D volume into smaller 3D patches (e.g., 96x96x96), runs the neural network on each patch, and seamlessly stitches the predictions back together into a complete full-brain segmentation mask.

### FastAPI, HTML, JavaScript
*   **FastAPI:** Used to build the high-performance backend API for the clinical dashboard. It asynchronously handles heavy model loading, image processing, and prediction routing.
*   **HTML & JavaScript:** Used in the frontend presentations and clinical dashboard web interfaces. JavaScript dynamically renders interactive UI elements like the 3D MRI slice viewers, risk gauges, and growth curve charts.

---

## 5. Datasets Used

1.  **Classification (Federated Learning):** 
    *   Masoud Brain Tumor MRI dataset
    *   BRISC 2025 dataset
    *   *Usage:* Partitioned to simulate 3 distinct hospital clients with varying levels of Non-IID skew (~9,300 total images).
2.  **Segmentation:** 
    *   BraTS 2021 dataset
    *   *Usage:* 1,251 patients with 4 MRI modalities each, used to train the 3D Attention U-Net.
3.  **Progression Forecasting:** 
    *   MU-Glioma-Post (TCIA)
    *   LUMIERE
    *   UCSD-PTGBM
    *   *Usage:* Longitudinal scans mapping tumor volumes over months/years to train the math and LSTM growth models.

---

## 6. Result Metrics

### Module 1: 3D Segmentation
*   **Mean Dice Score:** 0.76
*   **Tumor Core (TC) Dice:** 0.85
*   **Enhancing Tumor (ET) Dice:** 0.79
*   **Whole Tumor (WT) Dice:** 0.65

### Module 2: Federated Classification 
*(Setup 1 - Natural Heterogeneity)*
*   **FedAvg Final Accuracy:** 98.79% (Client Fairness $\sigma$: 1.58)
*   **FedProx Final Accuracy:** 99.29% (Client Fairness $\sigma$: 1.70)
*   **FedQPSO Final Accuracy:** 98.43% (**Client Fairness $\sigma$: 1.47 -> Best Fairness**)
*   **FedQPSO Glioma Recall:** 0.9593 (Highest, meaning fewest missed fatal diagnoses)
*(Under Label Skew, FedQPSO is the only method that maintains $\geq$80% accuracy for the weakest hospital).*

### Module 3: Progression Forecasting
*   **Math Baseline (Logistic) HGG MAE:** 24,672 mm³ ($R^2$ = 0.518)
*   **LSTM Hybrid HGG MAE:** 22,728 mm³ ($R^2$ = 0.592, an improvement of **+7.88%**)
*   The module accurately generates automated RANO clinical criteria (Complete Response, Partial Response, Stable Disease, Progressive Disease) and associated clinical risk alerts.
