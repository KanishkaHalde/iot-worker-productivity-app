import streamlit as st
import pandas as pd
import random
import time
import joblib

# Load the trained model and label encoders
model = joblib.load("productivity_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Maps for categorical features based on label encoders
quarter_map = {k: v for v, k in enumerate(label_encoders['quarter'].classes_)}
department_map = {k: v for v, k in enumerate(label_encoders['department'].classes_)}
day_map = {k: v for v, k in enumerate(label_encoders['day'].classes_)}

# Reverse maps for UI display
quarter_inv_map = {v: k for k, v in quarter_map.items()}
department_inv_map = {v: k for k, v in department_map.items()}
day_inv_map = {v: k for k, v in day_map.items()}

def generate_sensor_data():
    # Randomly generate categorical and numerical sensor data
    quarter = random.choice(list(quarter_map.values()))
    department = random.choice(list(department_map.values()))
    day = random.choice(list(day_map.values()))

    return {
        "quarter": quarter,
        "department": department,
        "day": day,
        "team": random.randint(1, 12),
        "targeted_productivity": round(random.uniform(0.5, 1.0), 2),
        "smv": round(random.uniform(10, 50), 2),
        "wip": random.randint(100, 3000),
        "over_time": random.randint(0, 1200),
        "incentive": random.randint(0, 200),
        "idle_time": random.randint(0, 180),
        "idle_men": random.randint(0, 15),
        "no_of_style_change": random.randint(0, 5),
        "no_of_workers": random.randint(10, 100)
    }

st.set_page_config(page_title="IoT Worker Productivity Dashboard", layout="wide")
st.title("📊 IoT Worker Productivity Dashboard")
st.markdown("Real-time simulation of worker productivity with IoT sensor data.")

placeholder = st.empty()

data_log = []

# Control simulation start/stop
if "running" not in st.session_state:
    st.session_state.running = False

def toggle_run():
    st.session_state.running = not st.session_state.running

start_stop = st.button("Start Simulation" if not st.session_state.running else "Stop Simulation", on_click=toggle_run)

simulation_cycles = st.number_input("Number of simulation cycles:", min_value=10, max_value=500, value=50)

if st.session_state.running:
    for _ in range(simulation_cycles):
        sensor_data = generate_sensor_data()

        # Convert to DataFrame for model input
        input_df = pd.DataFrame([sensor_data])

        # Predict productivity
        prediction = model.predict(input_df)[0]

        # Log data
        log_entry = sensor_data.copy()
        log_entry["predicted_productivity"] = prediction
        data_log.append(log_entry)

        with placeholder.container():
            st.subheader("📡 Latest IoT Sensor Data")
            # Show human-readable categories
            display_data = {
                "quarter": quarter_inv_map[sensor_data["quarter"]],
                "department": department_inv_map[sensor_data["department"]],
                "day": day_inv_map[sensor_data["day"]],
                **{k: v for k, v in sensor_data.items() if k not in ["quarter", "department", "day"]}
            }
            st.json(display_data)

            st.subheader("🎯 Predicted Productivity")
            st.metric(label="Prediction", value=f"{prediction:.3f}")

            st.subheader("📈 Productivity and WIP Over Time")
            hist_df = pd.DataFrame(data_log)
            if len(hist_df) > 1:
                st.line_chart(hist_df[["predicted_productivity", "wip"]])

        time.sleep(2)
else:
    st.info("Simulation paused. Click 'Start Simulation' to begin.")
