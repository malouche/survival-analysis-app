# Survival Analysis Calculator

A Streamlit app for estimating survival functions using both Kaplan-Meier and Nelson-Aalen methods.

## Features

- Input survival data with censoring indicators
- Calculate survival estimates using either Kaplan-Meier or Nelson-Aalen method
- Visualize survival curves
- Display survival function in LaTeX format
- Show results in a data table

## Installation

1. Clone this repository:
```bash
git clone https://github.com/malouche/survival-analysis-app.git
cd survival-analysis-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. In the web interface:
   - Select the estimation method (Kaplan-Meier or Nelson-Aalen)
   - Enter the number of observations
   - Input the time values and mark censored observations
   - Click "Calculate and Plot" to see the results

## Example

For example, you can input data like:
- Time 1: 5.0, Not censored
- Time 2: 6.0, Censored
- Time 3: 2.5, Not censored

The app will calculate and display:
- The survival curve plot
- The mathematical formula in LaTeX
- A table with the survival probabilities at each time point