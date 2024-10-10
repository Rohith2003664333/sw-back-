from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_cors import CORS
import joblib
import pandas as pd
import os
import math
from pymongo import MongoClient
import logging
from datetime import timedelta

# Initialize the app and setup CORS
app = Flask(__name__)
#CORS(app)
CORS(app, supports_credentials=True)


# Secret key for session management
app.secret_key = os.urandom(24)

# Set session to be permanent and set a longer session lifetime
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and data
cls = joblib.load('police_up.pkl')
en = joblib.load('label_encoder_up.pkl')
df1 = pd.read_csv('Sih_police_station_data.csv')
model = joblib.load('human_vs_animal.pkl')
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

df2['indicator'] = df2['total_crime_against_women'].apply(crime_indicator)

# MongoDB connection
client = MongoClient("mongodb+srv://rh0665971:q7DFaWad4RKQRiWg@cluster0.gusg4.mongodb.net/?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")
db = client['swaraksha']
users_collection = db['users']
messages_collection = db['messages']

@app.route('/community')
def community():
    username = session.get('username', 'Guest')  # Get the username from the session
    logger.info(f"Community page accessed by {username}.")
    return jsonify({"username": username})


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
    logger.info(f"Message sent by {username}: {data['message']}")
    return jsonify({"status": "Message sent!"})

# Route to get the username from the session
@app.route('/getUsername', methods=['GET'])
def get_username():
    username = session.get('username', 'Guest')
    return jsonify({"username": username})

# Route to handle SOS messages (triggered when SOS button is pressed)
@app.route('/sendSOS', methods=['POST'])
def send_sos():
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']
    address = data['address']
    username = session.get('username', 'Guest')
    mobile = session.get('mobile', 'Guest')
    sos_message = f"Emergency! Please help me at (address: {address}, Latitude: {latitude}, Longitude: {longitude}, mobile: {mobile})"
    new_message = {"message": sos_message, "username": username}
    messages_collection.insert_one(new_message)
    logger.info(f"SOS message sent by {username}: {sos_message}")
    return jsonify({"status": "SOS sent!"})

# Home page route
@app.route('/index')
def index():
    username = session.get('username', 'Guest')
    return render_template('index.html', username=username)

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash('All fields are required!')
            return redirect('register')

        # Check if email already exists
        if users_collection.find_one({'email': email}):
            flash('Email already exists! Please log in.')
            return jsonify({'success': True, 'message': 'Email already exists! Please log in.'})

        # Insert new user into MongoDB
        users_collection.insert_one({
            'username': username,
            'mobile': mobile,
            'email': email,
            'password': password  # Plain text for now as requested
        })

        flash('Registration successful! Please log in.')
        return jsonify({'success': True, 'message': 'Registration successful! Please log in.'})

    return render_template('registration.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if user exists
        user = users_collection.find_one({'email': email})

        if user and password == user['password']:
            session['username'] = user['username']
            session['mobile'] = user['mobile']
            session.permanent = True  # Set session as permanent
            logger.info(f"User {user['username']} logged in successfully.")
            return jsonify({'success': True, 'username': user['username']})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials!'})

    return render_template('login.html')

# Logout route to clear the session
@app.route('/logout', methods=['POST', 'GET'])
def logout():
    username = session.get('username', 'Guest')
    session.clear()  # Clear the session
    logger.info(f"User {username} logged out.")
    return redirect(url_for('index'))

# Route to find nearest police station using ML model
@app.route('/nearestPoliceStation', methods=['POST'])
def nearest_police_station():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    try:
        nearest_station = en.inverse_transform(cls.predict([[latitude, longitude]]))
        contact_number = df1.loc[df1['Police_station_name'].str.contains(nearest_station[0], case=False, na=False), 'phone_number'].values[0]
        contact_number = contact_number.replace('-', '')  # Clean number
        return jsonify({
            'police_station': nearest_station[0],
            'contact_number': contact_number
        })
    except Exception as e:
        logger.error(f"Error finding nearest police station: {e}")
        return jsonify({'error': str(e)})

# Additional emergency and utility routes (trimmed for brevity)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
