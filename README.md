# Traffic Accident Detection and Localization using Deep Learning

## Team 524 — Computer Vision Project

### Team Members
- Gagandeep
- Channing
- Viwesh
  
# Project Overview

This project focuses on traffic accident understanding from video data using deep learning and computer vision techniques. The main goal is to move beyond simple frame-level classification and build a system that can answer three important questions:

- What type of accident happened?
- When did the accident happen?
- Where did the accident happen?

To achieve this, we developed a multi-stage pipeline that progressively improves accident understanding.

The project pipeline includes:

1. **2D CNN** for frame-level accident classification  
2. **3D CNN** for spatio-temporal accident understanding  
3. **Temporal + Spatial Localization** using YOLO detections and bounding-box dynamics  

# Motivation

Traffic accidents are short and motion-dependent events. A single image frame may show vehicles and roads, but it often does not contain enough information to determine whether a collision has occurred.

Traditional image classification models are limited because they process frames independently and ignore temporal relationships. However, accident detection requires understanding how objects move and interact over time.

This project addresses that challenge by combining:
- spatial learning,
- temporal modeling,
- and localization.

# Dataset

This project uses the Kaggle ACCIDENT synthetic traffic collision dataset.

## Dataset Features

The dataset contains:
- Synthetic traffic accident videos
- Multiple accident classes
- Bounding-box annotations
- Accident timing information
- Sensor metadata

## Accident Classes

- head-on
- rear-end
- sideswipe
- single
- t-bone

## Important Observation

One important observation from exploratory data analysis is that accidents usually occur early in the video sequence.

- Mean accident time: **7.61 seconds**
- Median accident time: **6.90 seconds**

This observation influenced the design of the temporal modeling pipeline.

# Repository Structure

```text
Project-524---Computer-Vision/
│
├── Data_Preparation/
│   ├── Frame extraction
│   ├── Label preparation
│   └── Dataset preprocessing
│
├── EDA/
│   ├── Dataset visualization
│   ├── Timing distribution analysis
│   └── Annotation inspection
│
├── simple_CNN/
│   ├── CNN.py
│   ├── training scripts
│   └── inference pipeline
│
├── Viwesh-3DCNN/
│   ├── clip generation
│   ├── 3D CNN architecture
│   ├── training
│   ├── evaluation
│   └── CAM visualization
│
├── Temporal_Filter.ipynb
│   ├── temporal localization
│   └── spatial localization
│
└── README.md
