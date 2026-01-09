# Lightweight Brain Tumor Segmentation on Low-Resource Systems

This repository contains the code and materials from my implementation of the **"Lightweight Brain Tumor Segmentation on Low-Resource Systems"** tutorial by **MAILAB** (Medical Artificial Intelligence Lab).

The project focuses on building and training a lightweight deep learning model (based on a 3D U-Net architecture) for **automated brain tumor segmentation** from MRI scans, specifically optimized to run efficiently on **low-resource systems** (CPU-only, no GPU required).

It uses the **BraTS-Africa 2024** dataset — the first publicly available annotated brain MRI dataset from an African population, making it especially relevant for inclusive and generalizable medical AI in resource-constrained settings.

## Why this project matters

Developing models locally, with the end user in mind (especially in African healthcare environments with limited computing power), is incredibly important — as often highlighted by **Dr. Maruf Adewole**.  
This tutorial was a great hands-on experience that brought medical imaging machine learning closer to home.

## Key Learning Outcomes

Through this project, I gained practical experience in:

- Working with the **BraTS-Africa 2024** dataset (curated with support from SPARK and MAILAB)
- Developing and training models directly in **VS Code** (moving away from only using Google Colab/Jupyter)
- Using essential libraries for medical image processing and deep learning
- Understanding **U-Net** architecture and how it excels at segmentation tasks
- Optimizing models for **CPU-only** environments and low-capacity hardware
- Processing and working with **MRI** scans for the first time
- Evaluating segmentation performance using the **Dice score**
- Deploying the trained model with **Streamlit** for easy demo and sharing
- Building AI solutions with accessibility and real-world deployment in mind

## Project Structure
