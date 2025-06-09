# AI-Driven Intrusion Detection System (IDS)
This project is a real-time AI-driven Intrusion Detection System (IDS) that leverages machine learning algorithms to detect malicious network activity. It is designed to classify incoming traffic as benign or malicious, issue alerts, and provide a user-friendly dashboard for monitoring. The system is built using Python, Flask, Scikit-learn, and Socket.IO, with support for email alerts and asynchronous task handling via Celery.
---
## Features
-  Real-time network traffic analysis
-  Machine Learning-based threat classification (Random Forest)
-  Alert system with email notifications
-  Scalable backend using Flask and Celery
-  Interactive web dashboard with live updates (Socket.IO)
-  Batch upload support for traffic files (CSV, JSON)
-  Logging and historical threat tracking
---
## Project Structure
ai-ids/
│
├── app.py                             # Flask application (backend)
├── model/
│   ├── cicids_model.pkl        # Trained Random Forest model
│   └── cicids_scaler.pkl         # Feature scaler
│
├── templates/
│   └── index.html              # Frontend HTML dashboard
│
├── static/
│   └── styles.css              # CSS styles
│
├── uploads/                    # Uploaded network data (CSV/JSON)
├── requirements.txt            # Dependencies
├── server.log                  # Daily rotated logs
├── README.md                   # Project documentation
└── ...
---
## Prerequisites
- Python 3.8+
- pip
- Redis (for Celery task queue - optional)
- SMTP server (for email alerts)
---
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-ids.git 
   cd ai-ids
2. Install dependencies:
    pip install -r requirements.txt
3. Configure environment variables:
    Create a .env file:
    MAIL_SERVER=smtp.gmail.com
    MAIL_PORT=587
    MAIL_USE_TLS=True
    MAIL_USERNAME=your-email@gmail.com
    MAIL_PASSWORD=your-password
---
## Usage
1. Run the application:
    python app.py 
2. Access the dashboard:
    Open http://localhost:5000 in your browser.
3. Using the system:
    Real-time Monitoring: The dashboard displays live traffic analysis
    Manual Analysis: Submit network features via the input form
    Batch Processing: Upload CSV files containing network traffic data
    Alerts: View threats in the alerts panel and receive email notifications
---
## Machine Learning Model
    Algorithm: Random Forest (96.2% accuracy on test data)
    Dataset: Trained on CICIDS2017 dataset
    Features: 80+ network traffic features including:
Packet sizes
Protocol types
Flow duration
Source/destination statistics
---
## API Endpoints
  Method			Endpoint			Description
  POST			/predict			Submit traffic features for analysis
  GET			/logs			Fetch historical threat data
  POST			/upload			Upload CSV/JSON for batch analysis
  GET			/feature_importance			Retrieve top ML features
--- 
## Testing
    Run unit tests:
pytest tests/
---
## Deployment
    For production deployment:
    1. Use Gunicorn:
     gunicorn -w 4 -b 0.0.0.0:5000 app:app
    2. Set up Redis for Celery:
     celery -A app.celery worker --loglevel=info
    3. Configure a reverse proxy (Nginx/Apache)
---
## Troubleshooting
Common Issues:
   SMTP Errors: Ensure email credentials are correct and less secure apps are allowed
   Model Loading: Verify model files exist in model/ directory
   Performance: For high traffic, increase Celery workers and Gunicorn threads
---
## Contributing
1.	Fork the project
2.	Create your feature branch (git checkout -b feature/AmazingFeature)
3.	Commit your changes (git commit -m 'Add some AmazingFeature')
4.	Push to the branch (git push origin feature/AmazingFeature)
5.	Open a Pull Request
---
## License
This project is for academic and educational purposes only.
Contact Project Team if you want the file exported, published on GitHub, or tailored to a specific deployment environment (e.g., Docker, cloud hosting).
---
## Contributors
    Yahaya Sayeed – sayhah@icloud.com 
    Amissah Pearl Caroline – evatamakloe1@icloud.com 
    Dunyo Desmond Elikplim – desydunyo@gmail.com 
Supervisor: Rev. Dr. Albert A. Akanferi
University of Professional Studies, Accra — July 2025
