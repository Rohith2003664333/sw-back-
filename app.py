from flask import Flask, render_template, request, redirect, url_for, flash, jsonify,session
from flask_cors import CORS
import joblib
import pandas as pd
import os
import math
from pymongo import MongoClient
import logging
import numpy as np
import librosa
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
import sounddevice as sd
import soundfile as sf





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
app.secret_key = os.urandom(24)  # Ensure this is set for session handling

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# MongoDB connection
client = MongoClient("mongodb+srv://rh0665971:q7DFaWad4RKQRiWg@cluster0.gusg4.mongodb.net/?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")
db = client['swaraksha']
users_collection = db['users']
messages_collection = db.messages

# Route to render the community page
@app.route('/community')
def community():
    username = session.get('username', 'Guest')
    return render_template('community.html', username=username)

# Route to get all messages
@app.route('/getMessages', methods=['GET'])
def get_messages():
    messages = list(messages_collection.find({}, {'_id': 0, 'message': 1, 'username': 1}))
    messages_list = [{"message": msg.get('message', ''), "username": msg.get('username', 'Anonymous')} for msg in messages]
    return jsonify({"messages": messages_list})


# Route to send a message
@app.route('/sendMessage', methods=['POST'])
def send_message():
    data = request.json
    username = session.get('username', 'Guest')
    new_message = {"message": data['message'], "username": username}
    messages_collection.insert_one(new_message)
    return jsonify({"status": "Message sent!"})

# Route to handle SOS messages (triggered when SOS button is pressed)
@app.route('/sendSOS', methods=['POST'])
def send_sos():
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']
    address=data['address']
    username = session.get('username', 'Guest')
    mobile = session.get('mobile','Guest')
    sos_message = f"Emergency! Please help me at (address:{address},Latitude: {latitude}, Longitude: {longitude},mobile:{mobile})"
    new_message = {"message":sos_message, "username": username}
    messages_collection.insert_one(new_message)
    return jsonify({"status": "SOS sent!"})

# Home page route
@app.route('/index')
def home():
    
    username = session.get('username', 'Guest')
    return render_template('index.html',username=username)




@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Log request.form to see what data is being sent
        username = request.form.get('username')  # Use .get() to avoid KeyError
        mobile=request.form.get('mobile')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash('All fields are required!')
            return redirect('register')

        # Check if email already exists
        if users_collection.find_one({'email': email}):
            flash('Email already exists! Please log in.')
            return redirect('/')

        # Insert new user into MongoDB
        users_collection.insert_one({
            'username': username,
            'mobile':mobile,
            'email': email,

            'password': password  # Plain text for now as requested
        })

        flash('Registration successful! Please log in.')
        return redirect('/')

    return render_template('registration.html')



# Login route
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if user exists
        user = users_collection.find_one({'email': email})

        if user and password == user['password']:
            session['username'] = user['username']
            session['mobile']=user['mobile']
            flash('Login successful!')
            return redirect('index')  # Redirect to home after login
        else:
            flash('Invalid credentials! Please try again.')
            return redirect('/')
    
    return render_template('login.html')

# Logout route to clear the session
@app.route('/logout', methods=['POST', 'GET'])  # Specify allowed methods
def logout():
    session.clear()  # Clear the session
    flash('You have been logged out.')
    return redirect(url_for('index')) 

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
