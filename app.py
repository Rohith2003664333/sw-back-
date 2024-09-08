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
    print(f'Received emergency location: Latitude {latitude}, Longitude {longitude}, Address {address}')
    
    return jsonify({'status': 'success', 'latitude': latitude, 'longitude': longitude, 'address': address})

@app.route('/getCrimeAlert', methods=['GET'])
def get_crime_alert():
    city = request.args.get('city', '').lower()
    crime_alert = 'low'  # Default value
    for _, row in df2.iterrows():
        if city in row['registeration_circles'].lower():
            crime_alert = row['indicator']
            break
    return jsonify({'alert': crime_alert})

def record_audio(duration=20, sample_rate=44100):
    """Record audio for a given duration."""
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("🎤 Recording completed.")
    return audio

def extract_features(audio, sample_rate, n_segments=10):
    """Extract MFCC features from segmented audio."""
    segment_length = int(len(audio) / n_segments)
    mfcc_features = []
    
    for i in range(n_segments):
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=segment.flatten(), sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        test = mfcc_mean.reshape(1, -1)
        k = list(model.predict(test))
        
        if k[0] == 1:  # Check for human sound classification
            mfcc_features.append(mfcc_mean)
    
    return np.array(mfcc_features)

# Noise reduction
def noise_reduction(audio):
    """Apply basic noise reduction."""
    return librosa.effects.preemphasis(audio.flatten())

@app.route('/start_recording', methods=['POST'])
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
    print(f'Received emergency location: Latitude {latitude}, Longitude {longitude}, Address {address}')
    
    return jsonify({'status': 'success', 'latitude': latitude, 'longitude': longitude, 'address': address})

@app.route('/getCrimeAlert', methods=['GET'])
def get_crime_alert():
    city = request.args.get('city', '').lower()
    crime_alert = 'low'  # Default value
    for _, row in df2.iterrows():
        if city in row['registeration_circles'].lower():
            crime_alert = row['indicator']
            break
    return jsonify({'alert': crime_alert})

def record_audio(duration=20, sample_rate=44100):
    """Record audio for a given duration."""
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("🎤 Recording completed.")
    return audio

def extract_features(audio, sample_rate, n_segments=10):
    """Extract MFCC features from segmented audio."""
    segment_length = int(len(audio) / n_segments)
    mfcc_features = []
    
    for i in range(n_segments):
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=segment.flatten(), sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        test = mfcc_mean.reshape(1, -1)
        k = list(model.predict(test))
        
        if k[0] == 1:  # Check for human sound classification
            mfcc_features.append(mfcc_mean)
    
    return np.array(mfcc_features)

# Noise reduction
def noise_reduction(audio):
    """Apply basic noise reduction."""
    return librosa.effects.preemphasis(audio.flatten())

@app.route('/start_recording', methods=['POST'])
def start_recording():
    # Start audio recording
    audio = record_audio()
    audio = noise_reduction(audio)
    sf.write('sample.wav', audio, 44100)

    # Extract features
    features = extract_features(audio, 44100, n_segments=10)
    x = features

    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Perform hierarchical clustering
    clusters = sch.linkage(x_scaled, method='ward')
    max_d = 15  # Distance threshold for clustering
    cluster_labels = sch.fcluster(clusters, max_d, criterion='distance')

    # Estimate the number of people
    num_people = len(np.unique(cluster_labels))
    print(f"Estimated number of people: {num_people}")

    # Return number of people to the frontend
    return jsonify({'num_people': num_people})

if __name__ == '__main__':
    app.run(debug=True)
  

if __name__ == '__main__':
    app.run(debug=True)
