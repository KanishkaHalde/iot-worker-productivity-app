# IoT Worker Productivity Dashboard

This project is a real-time simulation and dashboard for predicting worker productivity in a manufacturing environment using IoT sensor data and machine learning.

Live app : https://iot-worker-appuctivity-app-9out4ejzip5upyhckughxa.streamlit.app/
## Overview

The app demonstrates how IoT sensors can collect real-time data from factory environments—such as temperature, machine utilization, idle time, and worker activity—and use a trained machine learning model to predict worker productivity live.

---

## Features

- Simulates IoT sensor data streaming continuously.
- Predicts worker productivity using a trained Random Forest regression model.
- Displays live sensor data and predicted productivity.
- Shows historical trends with dynamic line charts.
- Interactive controls to start/stop simulations and change cycle counts.

---

## Project Structure

- `train.py` - Data preprocessing, model training, evaluation, and saving the model and encoders.
- `app.py` - Streamlit web app for simulating sensor data, predicting productivity live, and displaying dashboards.
- `productivity_model.pkl` - Saved trained machine learning model.
- `label_encoders.pkl` - Saved label encoders for categorical features.
- `requirements.txt` - Python dependencies.

---

## Installation & Setup

1. Clone this repository:
