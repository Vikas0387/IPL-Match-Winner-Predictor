from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# -----------------------------
# Load model and dataset
# -----------------------------
model = joblib.load("model.pkl")
df = pd.read_csv("matches.csv")

# -----------------------------
# Prepare dropdown options
# -----------------------------
team_names = sorted(df['team1'].dropna().unique().tolist())
city_names = sorted(df['city'].dropna().unique().tolist())
venue_names = sorted(df['venue'].dropna().unique().tolist())

# Create label encodings (must match training)
team_mapping = {team: idx for idx, team in enumerate(team_names)}
city_mapping = {city: idx for idx, city in enumerate(city_names)}
venue_mapping = {venue: idx for idx, venue in enumerate(venue_names)}

# Reverse mapping for output
reverse_team_mapping = {v: k for k, v in team_mapping.items()}

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template(
        "index4.html",
        teams=team_mapping,
        cities=city_mapping,
        venues=venue_mapping
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Construct input array
    input_data = np.array([[
        int(data['city']),
        int(data['venue']),
        int(data['team1']),
        int(data['team2']),
        int(data['toss_winner']),
        int(data['toss_decision']),
        float(data['target_runs']),
        float(data['target_overs']),
        int(data['home_advantage_team1']),
        int(data['home_advantage_team2'])
    ]])

    # Make prediction
    prediction = int(model.predict(input_data)[0])
    predicted_team = reverse_team_mapping.get(prediction, "Unknown")

    return jsonify({
        "predicted_winner_encoded": prediction,
        "predicted_team": predicted_team
    })

# -----------------------------
# Run app
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
