# Brain_Sole
# Smart Running Shoe AI
### AI-Powered Running Performance and Injury Risk Analysis System

---

## Project Overview

Smart Running Shoe AI is an intelligent wearable system designed to monitor running biomechanics, analyze performance, assess injury risk, and provide personalized recommendations using Artificial Intelligence (AI) and Large Language Models (LLMs).

The system integrates Foot Pressure Sensors (FSR) and GPS technology to collect real-time running data. Machine Learning models analyze running patterns and predict potential injury risks, while an LLM-based coaching assistant provides personalized recommendations to improve running efficiency and reduce injury risk.

This project combines Sports Biomechanics, Embedded Systems, Machine Learning, and Mobile Application Development into a single smart running ecosystem.

---

# Problem Statement

Running is one of the most popular forms of exercise worldwide. However, many runners experience injuries caused by improper running mechanics, excessive training loads, and poor movement patterns.

Common running-related injuries include:

- Plantar Fasciitis
- Achilles Tendinopathy
- Shin Splints
- Patellofemoral Pain Syndrome
- Stress Fractures
- IT Band Syndrome

Most runners are unaware of these risks until symptoms become severe.

Therefore, an intelligent monitoring system capable of detecting abnormal running patterns and providing early intervention is needed.

---

# Objectives

The primary objectives of this project are:

1. Develop a smart running shoe capable of collecting biomechanical and GPS data.
2. Monitor foot pressure distribution during running.
3. Analyze running performance in real time.
4. Predict potential injury risks using Machine Learning.
5. Generate personalized recommendations using Large Language Models.
6. Develop a mobile application for visualization and monitoring.
7. Support injury prevention and performance optimization.

---

# Key Features

## Running Performance Analysis

- Running Speed
- Running Pace
- Distance Covered
- Cadence
- Stride Length
- Ground Contact Time
- Estimated Calories Burned

## Biomechanical Analysis

- Plantar Pressure Distribution
- Left-Right Balance
- Heel-Toe Loading Ratio
- Center of Pressure Analysis
- Running Symmetry Evaluation

## Injury Risk Assessment

The AI model classifies runners into:

- Low Risk
- Medium Risk
- High Risk

## AI Coaching Assistant

The LLM-based assistant can:

- Explain risk factors
- Suggest technique corrections
- Recommend training adjustments
- Provide recovery recommendations
- Answer user questions regarding running performance

---

# System Architecture

```text
FSR Sensors + GPS Module
            │
            ▼
      ESP32 Controller
            │
            ▼
      Data Acquisition
            │
            ▼
     Feature Extraction
            │
            ▼
 Machine Learning Model
            │
            ▼
 Injury Risk Prediction
            │
            ▼
     LLM Recommendation
            │
            ▼
 Mobile Application
