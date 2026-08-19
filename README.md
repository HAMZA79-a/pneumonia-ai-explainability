# Pneumonia AI Explainability

A medical image analysis project for **chest X-ray pneumonia classification** combining deep learning, visual explainability, and vision-language interpretation.

The pipeline integrates:

* **DenseNet121** for binary classification: `NORMAL` vs `PNEUMONIA`
* **Grad-CAM** for visual explanation of the CNN decision
* **MedGemma 1.5 4B** as a medical Vision-Language Model for textual interpretation of the Grad-CAM
* **Streamlit** for an interactive user interface
* **FastAPI + ngrok** to connect the local Streamlit application to MedGemma running on Google Colab GPU

## Pipeline

```text
Chest X-ray
     ↓
DenseNet121
     ↓
NORMAL / PNEUMONIA
     ↓
Prediction probability
     ↓
Grad-CAM
     ↓
X-ray + Grad-CAM + CNN prediction
     ↓
MedGemma VLM
     ↓
Natural-language explanation
     ↓
Streamlit interface
```

## Model

The classification model is based on **DenseNet121 pretrained on ImageNet**.

Training was performed in two phases:

1. Training of the classification head with the convolutional backbone frozen
2. Fine-tuning of the last DenseNet layers

The final classifier outputs the probability:

```text
P(PNEUMONIA)
```

A decision threshold of `0.75` is currently used in the application.

## Final test results

On the test set:

```text
Accuracy:     ~91%
NORMAL F1:    0.88
PNEUMONIA F1: 0.93
```

Final confusion matrix:

```text
                 Predicted
              NORMAL  PNEUMONIA

NORMAL          207       27
PNEUMONIA        27      363
```

## Grad-CAM Explainability

Grad-CAM is applied to the final convolutional representation of DenseNet121.

The feature maps used for the explanation have shape:

```text
7 × 7 × 1024
```

The Grad-CAM is generated for the **class actually predicted by the CNN**:

* for `PNEUMONIA`, the explanation is based on `P(PNEUMONIA)`
* for `NORMAL`, the explanation is based on `1 - P(PNEUMONIA)`

The resulting heatmap is superimposed on the original chest X-ray.

Because the feature map resolution is only `7 × 7`, the localization should be interpreted as **coarse visual attention**, not as precise lesion segmentation.

## Vision-Language Interpretation

The project uses **MedGemma 1.5 4B** to interpret the CNN visual attention.

MedGemma receives:

```text
Original chest X-ray
+
Grad-CAM overlay
+
CNN predicted class
+
CNN probability
```

The VLM is instructed to explain:

1. Main regions emphasized by Grad-CAM
2. Whether attention is located within the lung fields
3. Whether activation is focal, diffuse, unilateral, or bilateral
4. Whether attention is anatomically plausible for the CNN decision
5. Whether the CNN may rely on irrelevant structures or image borders
6. The main uncertainty of the Grad-CAM explanation

The VLM is explicitly instructed **not to independently diagnose the X-ray** and not to override the CNN classification.

## Streamlit Application

The Streamlit interface allows the user to:

* upload a chest X-ray
* obtain the DenseNet121 prediction
* visualize the prediction probability
* visualize the Grad-CAM explanation
* obtain a MedGemma interpretation of the model attention

Run locally with:

```bash
streamlit run app.py
```

## MedGemma API

MedGemma is executed on a **Google Colab GPU**.

A FastAPI endpoint exposes the VLM through an ngrok tunnel:

```text
Streamlit
    ↓
POST /interpret
    ↓
FastAPI on Google Colab
    ↓
MedGemma
    ↓
Interpretation returned to Streamlit
```

The ngrok URL is temporary and must be updated when the Colab session is restarted.

For security reasons, authentication tokens and secrets are **not included in this repository**.

## Project Structure

```text
pneumonia-ai-explainability/
│
├── app.py
├── requirements.txt
├── best_densenet121_phase2.keras
├── finale_file.ipynb
├── Vlam.ipynb
├── .gitignore
└── README.md
```

### Files

`app.py`
Streamlit application combining classification, Grad-CAM and VLM interpretation.

`finale_file.ipynb`
DenseNet121 training, evaluation and Grad-CAM explainability pipeline.

`Vlam.ipynb`
MedGemma VLM experiments and FastAPI integration.

`best_densenet121_phase2.keras`
Fine-tuned DenseNet121 model.

`requirements.txt`
Python dependencies required by the project.

## Installation

Clone the repository:

```bash
git clone https://github.com/HAMZA79-a/pneumonia-ai-explainability.git
cd pneumonia-ai-explainability
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Important Note

This project is intended for **research and educational purposes**.

The system is an AI-assisted image analysis prototype and **does not replace interpretation by a qualified healthcare professional**.

Grad-CAM visualizations indicate model attention and should not be interpreted as precise lesion localization.

## Technologies

`Python` · `TensorFlow` · `Keras` · `DenseNet121` · `Grad-CAM` · `MedGemma` · `Transformers` · `Streamlit` · `FastAPI` · `ngrok`

## Author

Developed as part of an AI / medical imaging project.
