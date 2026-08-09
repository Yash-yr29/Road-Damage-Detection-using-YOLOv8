# Road Damage Detection using YOLO

## 1. PROJECT DESCRIPTION

This project focuses on detecting and classifying **road damage**, such as potholes and cracks, from road images using a **pretrained YOLO (You Only Look Once) object-detection model**.

The project covers the complete object-detection workflow, including **dataset preparation, model fine-tuning, checkpointing, validation, performance evaluation, and inference**.

Model performance is evaluated using metrics such as **Precision, Recall, F1-Score, mAP@50, and mAP@50–95**.

---

## 2. FEATURES

- **Road damage detection** from images using YOLO
- Fine-tuned a **pretrained YOLO object-detection model** on the **RDD2022 dataset**
- Detects and classifies **5 types of road damage**:
  - **D00** — Longitudinal Crack
  - **D10** — Transverse Crack
  - **D20** — Alligator Crack
  - **D40** — Pothole
  - **D50** — Repair-related damage
- **Bounding-box based object detection**
- Training and validation using separate datasets
- **Model checkpoint saving** during training
- Supports continuing fine-tuning from **saved model weights**
- Performance evaluation using:
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **mAP@50**
  - **mAP@50–95**
- Generates **training results, confusion matrices, and evaluation curves**
- Supports **inference on unseen road images**

---

## 3. DATASET

The dataset used in this project is the **RDD2022 (Road Damage Dataset 2022)** obtained from Kaggle.

RDD2022 is a large-scale, multinational road-surface image dataset designed to support the development of **AI-powered road infrastructure inspection systems**.

The dataset contains road images with **bounding-box annotations** for different types of road damage.

This project uses the following **5 damage categories**:

| Class | Description |
|------|-------------|
| D00 | Longitudinal Crack |
| D10 | Transverse Crack |
| D20 | Alligator Crack |
| D40 | Pothole |
| D50 | Repair-related Damage |

The dataset was divided into **training, validation, and testing data** for model development and evaluation.

---

## 4. TECH STACK

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **PyTorch**
- **Ultralytics YOLO**
- **Streamlit**
- **Google Colab**
- **Google Drive**

---

## 5. WORKFLOW

- **Loading Data**
- **Understanding the Dataset**
- **Preparing Bounding-Box Annotations**
- **Training / Fine-Tuning the YOLO Model**
- **Model Validation**
- **Performance Evaluation**
- **Inference**
- **Deployment**

---

## 6. RESULTS

The model achieved the following performance on the validation dataset:

| Metric | Score |
| :--- | ---: |
| **Precision** | **0.598** |
| **Recall** | **0.514** |
| **F1-Score** | **0.553** |
| **mAP@50** | **0.540** |
| **mAP@50–95** | **0.281** |

These results demonstrate that the fine-tuned YOLO model is capable of detecting and classifying different types of road damage from images.

---

## 7. CHALLENGES

- Limited **computational/GPU resources** during model training.
- Handling **Google Colab GPU/runtime disconnections** during long training sessions.
- Saving **model checkpoints at every epoch** to prevent loss of training progress.
- Continuing experiments from previously saved **model weights**.
- Managing **dataset paths and configuration files** across Google Colab and Google Drive.
- Comparing model performance across different training experiments.

---

## 8. WHAT I LEARNED

- How **object detection** works using YOLO.
- How to prepare and configure a **custom dataset** for YOLO.
- How **bounding-box annotations** are used for object detection.
- How to **fine-tune a pretrained YOLO model** on a custom dataset.
- How to save and manage **model checkpoints** during training.
- How to evaluate an object-detection model using **Precision, Recall, F1-Score, mAP@50, and mAP@50–95**.
- How to handle **interrupted training** and continue experiments using saved model weights.
- How to analyse **training and validation performance**.
- How to deploy a trained model using **Streamlit**.

---

## 9. DEPLOYMENT LINK

**Coming soon...**

---

## 10. FUTURE IMPROVEMENTS

- Experiment with different **YOLO architectures and hyperparameters** to improve performance.
- Increase **mAP@50 and mAP@50–95** through better training and fine-tuning strategies.
- Experiment with different **image sizes, batch sizes, learning rates, and optimizers**.
- Use stronger **data augmentation techniques** to improve model generalization.
- Optimize the model for **faster inference and lower computational requirements**.
- Test the model on **real-world road images** and different environmental conditions.
- Improve the **Streamlit interface** for a better user experience.