from flask import Flask, request, render_template, jsonify, send_from_directory, abort
from flask_socketio import SocketIO, emit
from email.mime.text import MIMEText
from celery import Celery
import redis
import os
import smtplib
from datetime import datetime
import os
import logging
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Redirect all logs to server.log file
file_handler = logging.FileHandler('server.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
import pandas as pd
import joblib
import numpy as np
from werkzeug.utils import secure_filename
import threading
import time
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Configure logging
logging.basicConfig(filename='server.log', level=logging.INFO,
                   format='%(asctime)s %(levelname)s: %(message)s')

# Email Configuration (replace with your credentials)
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', 'teenlordz@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'hegz mljd kjol agrl')  # Use App Password for Gmail

# Initialize SocketIO
socketio = SocketIO(app)

# Load ML model and scaler - FIXED PATHS
model = joblib.load('model/cicids_model.pkl')
scaler = joblib.load('model/cicids_scaler.pkl')

# Threat severity classification
SEVERITY_THRESHOLDS = {
    'Critical': 0.95,
    'High': 0.85,
    'Medium': 0.70,
    'Low': 0.50
}

# Initialize Celery
def make_celery(app):
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    celery = Celery(
        app.import_name,
        backend=redis_url,
        broker=redis_url
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

@celery.task
def send_alert_email_task(threat_details):
    logging.info("send_alert_email_task called with threat_details")
    try:
        msg = MIMEText(f"Threat detected at {datetime.now()}:\n\n{threat_details}")
        msg['Subject'] = "🚨 IDS Alert: Malicious Activity Detected"
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = "teenlordz@gmail.com"

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        logging.info("Alert email sent successfully")
        return True
    except Exception as e:
        logging.error(f"Email failed: {e}")
        return False

def classify_threat(confidence):
    """Determine threat severity based on confidence score"""
    for severity, threshold in SEVERITY_THRESHOLDS.items():
        if confidence >= threshold:
            return severity
    return 'Info'

@app.route('/')
def dashboard():
    """Render main dashboard"""
    return render_template('index.html')

from model.feature_processor import FeatureProcessor

# Initialize processor
feature_processor = FeatureProcessor()
feature_processor.load_scaler('model/cicids_scaler.pkl')

@app.route('/feature_importance')
def feature_importance():
    """Return feature importance for guidance"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importance = model.named_steps['classifier'].feature_importances_
    else:
        return jsonify({'error': 'Feature importance not available'}), 400

    features = feature_processor.required_features
    return jsonify({
        'most_important': [
            {'feature': f, 'importance': float(i)}
            for f, i in sorted(zip(features, importance),
                             key=lambda x: x[1],
                             reverse=True)[:10]
        ],
        'required_features': features
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle real-time prediction requests"""
    # Check Content-Type header
    if request.content_type != 'application/json':
        return jsonify({'error': "Unsupported Media Type: Content-Type must be 'application/json'"}), 415

    try:
        # Get data from request
        data = request.get_json()

        # Validate input
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid input format'}), 400

        # Validate feature length
        if len(data['features']) != len(feature_processor.required_features):
            return jsonify({'error': f"Feature length mismatch: expected {len(feature_processor.required_features)}, got {len(data['features'])}"}), 400

        # Convert features list to DataFrame for preprocessing
        df_features = pd.DataFrame([data['features']], columns=feature_processor.required_features)

        # Preprocess features
        df_processed = feature_processor.preprocess(df_features)

        # Scale features
        scaled_features = scaler.transform(df_processed)

        # Make prediction
        prediction = model.predict(scaled_features)
        proba = model.predict_proba(scaled_features)
        confidence = np.max(proba)

        # Determine threat type and severity
        is_threat = prediction[0] == 1
        threat_type = "Malicious" if is_threat else "Benign"
        severity = classify_threat(confidence) if is_threat else "None"

        # Prepare response
        result = {
            'threat': bool(is_threat),
            'threat_type': threat_type,
            'severity': severity,
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat()
        }

        # If threat detected, log and alert
        if is_threat:
            threat_id = datetime.now().strftime("%Y%m%d%H%M%S")
            alert_data = {
                'threat_id': threat_id,
                'threat_type': threat_type,
                'severity': severity,
                'confidence': confidence,
                'source_ip': data.get('source_ip', 'Unknown'),
                'description': data.get('description', 'No description'),
                'timestamp': result['timestamp']
            }

            # Log threat
            logging.warning(f"Threat detected: {alert_data}")

            # Send real-time alert
            socketio.emit('new_alert', alert_data)

            # Send email alert asynchronously via Celery
            send_alert_email_task.delay(str(alert_data))

        return jsonify(result)

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle batch file upload for analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Process uploaded file
            df = pd.read_csv(filepath)
            results = []

            for _, row in df.iterrows():
                # Convert row to DataFrame for preprocessing
                df_row = pd.DataFrame([row.values], columns=df.columns)

                # Validate feature length
                if len(df_row.columns) != len(feature_processor.required_features):
                    return jsonify({'error': f"Feature length mismatch in uploaded file: expected {len(feature_processor.required_features)}, got {len(df_row.columns)}"}), 400

                df_processed = feature_processor.preprocess(df_row)

                scaled_features = scaler.transform(df_processed)
                prediction = model.predict(scaled_features)
                proba = model.predict_proba(scaled_features)
                confidence = np.max(proba)

                results.append({
                    'input': row.to_dict(),
                    'threat': bool(prediction[0] == 1),
                    'confidence': float(confidence),
                    'severity': classify_threat(confidence) if prediction[0] == 1 else 'None'
                })

            return jsonify({'results': results})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'File type not allowed'}), 400

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv', 'json'}

@app.route('/files', methods=['GET'])
def list_files():
    """List files in the uploads directory"""
    try:
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        # Include server.log in the list if it exists
        if os.path.exists('server.log'):
            files.append('server.log')
        return jsonify({'files': files})
    except Exception as e:
        logging.error(f"Error listing files: {str(e)}")
        return jsonify({'error': 'Could not list files'}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a file from the uploads directory or root for server.log"""
    try:
        # Secure the filename
        if filename == 'server.log':
            if not os.path.isfile('server.log'):
                abort(404)
            return send_from_directory('.', 'server.log', as_attachment=True)
        else:
            if not os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
                abort(404)
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        logging.error(f"Error downloading file {filename}: {str(e)}")
        return jsonify({'error': 'Could not download file'}), 500

if __name__ == '__main__':
    import socket
    import time

    # Temporary test endpoint for email sending
    @app.route('/test_email')
    def test_email():
        test_data = {
            'threat_id': 'test123',
            'threat_type': 'Malicious',
            'severity': 'High',
            'confidence': 0.99,
            'source_ip': '192.168.1.1',
            'description': 'Test email alert',
            'timestamp': datetime.now().isoformat()
        }
        send_alert_email_task.delay(str(test_data))
        return "Test email task triggered. Check your email and logs."

    # Create upload directory if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    def generate_live_traffic():
        protocols = ['TCP', 'UDP', 'ICMP', 'Other']
        threat_levels = ['None', 'Low', 'Medium', 'High']
        while True:
            entry = {
                'id': 'traffic-' + str(int(time.time() * 1000)),
                'timestamp': time.strftime('%H:%M:%S'),
                'srcIp': f"192.168.{random.randint(0,254)}.{random.randint(0,254)}",
                'dstIp': f"10.0.{random.randint(0,254)}.{random.randint(0,254)}",
                'protocol': random.choice(protocols),
                'port': random.randint(1, 65535),
                'size': random.randint(40, 1500),
                'duration': random.randint(1, 500),
                'threatLevel': random.choice(threat_levels)
            }
            socketio.emit('live_traffic', entry)
            time.sleep(random.uniform(0.5, 2.5))

    # Start background thread for live traffic simulation
    thread = threading.Thread(target=generate_live_traffic)
    thread.daemon = True
    thread.start()

    # Start the application with port conflict handling
    def is_port_in_use(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('0.0.0.0', port)) == 0

    start_port = int(os.getenv('FLASK_RUN_PORT', 5000))
    max_tries = 5
    port = start_port
    for i in range(max_tries):
        if not is_port_in_use(port):
            try:
                print(f"Starting server on port {port}")
                socketio.run(app, debug=True, host='0.0.0.0', port=port)
                break
            except OSError as e:
                if e.errno == 10048:  # Address already in use
                    print(f"Port {port} in use, trying next port")
                    port += 1
                    time.sleep(1)
                else:
                    raise
        else:
            print(f"Port {port} already in use, trying next port")
            port += 1
            time.sleep(1)
    else:
        print(f"Could not start server after {max_tries} attempts. Please free a port and try again.")
