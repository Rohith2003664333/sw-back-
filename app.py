from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import librosa
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import soundfile as sf
import pandas as pd
from werkzeug.utils import secure_filename
import os
import logging
import math

cls = joblib.load('police_up.pkl')
en = joblib.load('label_encoder_up.pkl')  

df1=pd.read_csv('Sih_police_station_data.csv')


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained model
model = joblib.load('human_vs_animal.pkl')

# Load crime data
df2 = pd.read_csv('districtwise-crime-against-women (1).csv')
df2 = df2[['registeration_circles', 'total_crime_against_women']]

# Define function to classify crime alert
def crime_indicator(crime_count):
    if crime_count < 50:
        return '🟢Green'
    elif 50 <= crime_count <= 500:
        return '🟡Yellow'
    else:
        return '🔴Red'

# Apply classification to crime data
df2['indicator'] = df2['total_crime_against_women'].apply(crime_indicator)

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emergency_contacts')
def emergency_contacts():
    return render_template('emergency_contacts.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/nearestPoliceStation', methods=['POST'])
def nearest_police_station():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    # Predict the nearest police station using the trained model
    try:
        nearest_police_station.nearest_station = en.inverse_transform(cls.predict([[latitude, longitude]]))
        contact_number = df1.loc[df1['Police_station_name'].str.contains(nearest_police_station.nearest_station[0], case=False, na=False), 'phone_number'].values[0]
        n = contact_number.replace('-', '')  # Clean number
        return jsonify({
            'police_station': nearest_police_station.nearest_station[0],
            'contact_number': n  # Ensure you return the cleaned number
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/distanceP', methods=['POST'])
def distance_p():
    data = request.get_json()
    lat1 = data.get('latitude')
    lon1 = data.get('longitude')
    nearest_station = en.inverse_transform(cls.predict([[lat1, lon1]]))[0]
    lat1=float(lat1)
    lon1=float(lon1)

    # Get the nearest station name and location
    
    station_data = df1[df1['Police_station_name'].str.contains(nearest_station, case=False, na=False)]

    lat2 = station_data['latitude'].values[0]
    lon2 = station_data['longitude'].values[0]

    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    R = 6371000  # Earth's radius in meters
    distance = (R * c)/1000
    distance = round(distance,2)

    return jsonify({'police_distance': distance})

    

              
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emergency', methods=['POST'])
def emergency():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    address = data.get('address')
    
    # Log received location and address
    logger.info(f'Received emergency location: Latitude {latitude}, Longitude {longitude}, Address {address}')
    
    return jsonify({'status': 'success', 'latitude': latitude, 'longitude': longitude, 'address': address})

@app.route('/getCrimeAlert', methods=['GET'])
def get_crime_alert():
    city = request.args.get('city')
    crime_alert = 'low'  # Default value
    for i in range(len(df2)):
        if city.lower() in df2['registeration_circles'][i].lower():
            crime_alert = df2['indicator'][i]
            break
    return jsonify({'alert': crime_alert})

if __name__ == '__main__':
    app.run(debug=True)
