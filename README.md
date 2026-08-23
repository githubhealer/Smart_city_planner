#UrbanSense AI

An urban decision-support system for the Exasol AI Build Hackathon. The project combines heterogeneous city data such as weather, traffic, and air quality to identify emerging urban problems, estimate their potential impact, and recommend actionable interventions.

The current implementation focuses on the weather component and demonstrates the complete data-to-decision pipeline using Milan weather data and machine learning.

## Features

* Weather data processing and daily aggregation
* Rain prediction using machine learning
* Logistic Regression baseline
* Random Forest final model
* Rain probability prediction using Exasol Python UDFs
* Rain-risk classification
* City action recommendations based on predicted risk
* Planned integration of traffic and air-quality data

## Technologies

* Exasol
* SQL
* Python
* Exasol Python UDFs
* NumPy
* scikit-learn
* Logistic Regression
* Random Forest

## Urban Datasets

The hackathon solution uses three main data sources.

### Weather

* Temperature
* Humidity
* Dew point
* Precipitation
* Rain
* Pressure
* Cloud cover
* Wind
* Solar radiation

### Traffic

Traffic data will be used to identify and assess traffic conditions and their interaction with other urban factors.

Integration: To be added

### Air Quality

Air-quality data will be used to identify pollution-related conditions and their interaction with weather and traffic.

Integration: To be added

The current implementation has completed the weather-based prediction pipeline. Traffic and air-quality integration are part of the overall solution and are the next components of the system.

## Current ML Pipeline

Weather Data → Daily Aggregation → Feature Engineering → Exasol → Random Forest → Rain Probability → Rain Risk → City Action

The planned complete urban-data pipeline is:

Weather + Traffic + Air Quality → Urban Risk Analysis → Impact Prediction → Recommended Action

## Model

The current Random Forest model uses 16 weather features.

Configuration:

* n_estimators = 300
* max_depth = 8
* min_samples_leaf = 3
* random_state = 42

The model produces both a rain probability and a rain/no-rain prediction.

## Results

The current test set contains 363 days.

Random Forest confusion matrix:

| Actual  | Predicted | Days |
| ------- | --------- | ---: |
| No Rain | No Rain   |  157 |
| No Rain | Rain      |   93 |
| Rain    | No Rain   |   35 |
| Rain    | Rain      |   78 |

Accuracy: 64.74%

Rain probabilities ranged from approximately 4.06% to 67.74%.

## Risk Classification

| Probability | Risk      | City Action                             |
| ----------- | --------- | --------------------------------------- |
| < 20%       | LOW       | Normal city operations                  |
| 20%–<40%    | MODERATE  | Monitor drainage and outdoor activities |
| 40%–<60%    | HIGH      | Prepare drainage and traffic management |
| >= 60%      | VERY HIGH | Activate high-rain preparedness         |

Current test-set distribution:

* LOW: 84 days
* MODERATE: 183 days
* HIGH: 89 days
* VERY HIGH: 7 days

## Exasol Integration

The current machine-learning implementation runs inside Exasol using Python UDFs.

The UDF-based pipeline performs model training and prediction, producing rain probabilities that are converted into city-risk information.

## How to Run

Requires:

* Exasol database
* Python 3
* NumPy
* scikit-learn

Detailed setup and execution instructions: To be added

## Project Structure

project/
├── README.md
├── udf/
│   └── Exasol Python UDF scripts
├── sql/
│   └── SQL scripts
├── data/
│   └── Dataset information
├── presentation/
│   └── Pitch deck
└── demo/
└── Demo video link

## Hackathon Problem

Cities generate large amounts of data from traffic, air quality, weather, public transportation, accidents, and other urban services. This data is often distributed across different datasets and mainly used for monitoring rather than timely, data-driven decisions.

The challenge is to use heterogeneous urban data to automatically identify emerging problems, predict their potential impact, and recommend actionable interventions to city authorities.

Our current implementation establishes the weather prediction and decision-support component. The next stage is to combine it with traffic and air-quality information to produce a broader urban-risk assessment.

## Next Steps

* Integrate traffic dataset
* Integrate air-quality dataset
* Combine weather, traffic, and air-quality signals
* Identify cross-domain urban problems
* Estimate potential impact
* Generate combined city-level recommendations

## Team

Sandeep Ganesh Deepak Ganesh
Rithikha S
Jeffrey Navin
Swetha Arul
