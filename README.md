# UrbanSense AI

UrbanSense AI is an urban decision-support system developed for the Exasol AI Build Hackathon. It combines machine learning, Exasol, and AI to transform weather and air-quality data into practical, data-driven recommendations.

The current implementation focuses on weather and air quality and demonstrates the complete pipeline from raw urban data to machine-learning predictions and AI-powered decisions.

## Problem

Cities generate large volumes of data from different urban systems such as weather and air quality. This data is often processed independently and primarily used for monitoring rather than making timely, actionable decisions.

For example:

- Rain can affect commuting and outdoor activities.
- Poor air quality can create health and environmental risks.
- Different urban signals can indicate risks that are difficult to interpret from raw data alone.

The challenge is to transform this heterogeneous urban data into meaningful predictions, identify potential risks, and provide actionable recommendations.

## Solution

UrbanSense AI addresses this problem through a three-layer architecture:

1. **Machine Learning Layer**  
   Weather and air-quality data are processed and used to train prediction models.

2. **Exasol Data and Prediction Layer**  
   Exasol stores the processed data and machine-learning predictions and provides the data layer for the AI application.

3. **AI Decision-Support Layer**  
   Amazon Nova interprets natural-language user requests, retrieves the relevant predictions through a tool connected to Exasol, and generates practical recommendations.

The overall pipeline is:

```text
Urban Data
    ↓
Data Preprocessing
    ↓
Machine Learning
    ↓
Exasol
    ↓
Predictions
    ↓
Amazon Nova
    ↓
Actionable Recommendation
