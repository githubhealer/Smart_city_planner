# Milan Smart City Rain Risk Prediction

A weather-based decision-support component for the Exasol AI Build Hackathon. The project uses historical Milan weather data to predict rain probability and convert it into actionable rain-risk levels for city planning.

## Features

- Daily weather data preparation from hourly observations
- Rain prediction using machine learning
- Logistic Regression as a baseline model
- Random Forest as the final model
- Rain probability prediction using Exasol Python UDFs
- Rain-risk classification:
  - Low
  - Moderate
  - High
  - Very High
- City action recommendations based on predicted risk

## Technologies

- Exasol
- SQL
- Python
- Exasol Python UDFs
- NumPy
- scikit-learn
- Random Forest
- Logistic Regression

## Dataset

The current implementation uses the **weather dataset** for Milan.

The original hourly dataset contains:

- Temperature
- Relative humidity
- Dew point
- Apparent temperature
- Precipitation
- Rain
- Pressure
- Cloud cover
- Wind speed
- Wind direction
- Wind gusts
- Shortwave radiation

The data was aggregated from hourly observations to daily features.

Traffic and air-quality datasets are part of the overall hackathon problem statement but have not yet been integrated into the current implementation.

## Model

The final Random Forest model uses 16 weather features.

Configuration:

- `n_estimators = 300`
- `max_depth = 8`
- `min_samples_leaf = 3`
- `random_state = 42`

The model generates a rain probability and a rain/no-rain prediction.

## Results

The test set contains 363 days.

Random Forest confusion matrix:

| Actual | Predicted | Days |
|---|---|---:|
| No Rain | No Rain | 157 |
| No Rain | Rain | 93 |
| Rain | No Rain | 35 |
| Rain | Rain | 78 |

Accuracy: **64.74%**

The model's rain probabilities ranged from approximately **4.06% to 67.74%**.

## Risk Classification

| Probability | Risk | City Action |
|---|---|---|
| < 20% | LOW | Normal city operations |
| 20%–<40% | MODERATE | Monitor drainage and outdoor activities |
| 40%–<60% | HIGH | Prepare drainage and traffic management |
| >= 60% | VERY HIGH | Activate high-rain preparedness |

Current test-set distribution:

- LOW: 84 days
- MODERATE: 183 days
- HIGH: 89 days
- VERY HIGH: 7 days

## How to Run

Requires:

- Exasol database
- Python 3
- NumPy
- scikit-learn

The current ML implementation is executed through Exasol Python UDFs.

Detailed setup and execution instructions: **[To be added]**

## Project Structure

```text
project/
├── README.md
├── udf/
│   └── [Exasol Python UDF scripts]
├── sql/
│   └── [SQL scripts]
├── data/
│   └── [Dataset information]
├── presentation/
│   └── [Pitch deck]
└── demo/
    └── [Demo video link]
