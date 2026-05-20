import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the original dataset
df = pd.read_csv('ev_fleet_telemetry_synthetic.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Minute'] = df['Timestamp'].dt.minute
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

df = df.dropna()
df = df.sort_values(by=['Vehicle_ID', 'Timestamp'])

# Feature Engineering (Lag Features)
df['Prev_SoC'] = df.groupby('Vehicle_ID')['SoC (%)'].shift(1)
df['Prev_Speed'] = df.groupby('Vehicle_ID')['Speed (Km/h)'].shift(1)
df['Prev_Acceleration'] = df.groupby('Vehicle_ID')['Acceleration (m/s²)'].shift(1)
df = df.dropna()

# Define Reduced Features
target = 'SoC (%)'
categorical_cols = ['Terrain', 'Weather']
features = ['Speed (Km/h)', 'Terrain', 'Weather', 'Prev_SoC', 'Braking (m/s²)', 'Acceleration (m/s²)']
numerical_cols = ['Speed (Km/h)', 'Prev_SoC', 'Braking (m/s²)', 'Acceleration (m/s²)']

X = df[features]
y = df[target]

# Preprocessing Pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search for Hyperparameter Tuning
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [5, 10, 15, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': [None, 'sqrt', 'log2']
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Model Prediction
y_pred = best_model.predict(X_test)


# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n📊 MAE (SoC Prediction): {mae:.2f}")
print(f"📉 RMSE (SoC Prediction): {rmse:.2f}")

# Save the entire pipeline, including preprocessing and model
joblib.dump(best_model, 'ev_range_predictor_reduced.pkl')
print("✅ Entire reduced pipeline saved as ev_range_predictor_reduced.pkl")