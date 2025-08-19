import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

# Load data
data = pd.read_csv('newpropro.csv')

# Clean data
data = data.dropna(subset=[
    'quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv', 'wip',
    'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers',
    'actual_productivity'
])

# Encode categorical features
label_encoders = {}
for col in ['quarter', 'department', 'day']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Features & target
features = [
    'quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv', 'wip',
    'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers'
]
X = data[features]
y = data['actual_productivity']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f'R2 Score: {r2_score(y_test, y_pred):.3f}')

# Save model and encoders
joblib.dump(model, 'productivity_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
