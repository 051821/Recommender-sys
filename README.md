Restaurant Recommender Web Application
This project is a Flask-based web application that recommends restaurants based on user inputs such as city, cuisine, country, cost, and rating. It uses a machine learning pipeline for recommendations.

Features
User-friendly web interface

ML-based restaurant recommendations

Results shown as an interactive table

Docker support for easy deployment

Project Structure
bash
Copy
Edit
/app
  ├── app.py                # Flask application
  ├── model_utiles.py       # ML model and recommendation logic
  ├── templates/
  │    ├── index.html       # Input form
  │    └── results.html     # Recommendation results
  ├── requirements.txt      # Python dependencies
Dockerfile                 # Docker image setup
docker-compose.yml         # Docker Compose config
README.md                 # Project documentation
How to Run Locally
Install dependencies:

nginx
Copy
Edit
pip install -r requirements.txt
Run the Flask app:

nginx
Copy
Edit
python app.py
Open browser at http://127.0.0.1:5000

How to Run with Docker
Build the Docker image:

nginx
Copy
Edit
docker build -t restaurant-recommender:latest .
Run the Docker container:

arduino
Copy
Edit
docker run -p 5000:5000 restaurant-recommender:latest
Visit http://127.0.0.1:5000 in your browser.

How to Run with Docker Compose
Run:

css
Copy
Edit
docker-compose up --build
Visit http://127.0.0.1:5000

Usage
Enter your preferences on the home page form.

Submit to get restaurant recommendations.

Navigate back to input new preferences.

Requirements
Flask

pandas

scikit-learn

jinja2

Notes
Ensure model_utiles.py correctly loads your ML model and data.

Debug mode is on in Flask; disable for production.

Modify Docker files if adding other services.

