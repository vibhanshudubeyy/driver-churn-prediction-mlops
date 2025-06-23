# Driver Churn Prediction MLOps Project

An MLOps pipeline to predict driver churn for a logistics company like Porter, built for a Data Engineering Intern application. The project uses Python, scikit-learn, FastAPI, MySQL, and MLflow to process driver data, train a Random Forest model, deploy predictions via a REST API, and monitor performance.

## Features
- ETL pipeline to process driver data (CSV to MySQL).
- Random Forest model for churn prediction (F1-score ≥ 0.80).
- FastAPI endpoint for real-time predictions.
- MySQL storage for predictions.
- MLflow for experiment tracking.
- Basic data drift monitoring.

## Tech Stack
- Python: pandas, scikit-learn, FastAPI
- Database: MySQL
- ML Tools: MLflow
- API: FastAPI, Uvicorn
- Scheduling: Python schedule
- Containerization: Docker
- Version Control: Git

## Setup Instructions
1. **Install Prerequisites**:
   - Python 3.9, MySQL, Docker, Git
   - Clone the repository:
     ```bash
     git clone https://github.com/vibhanshudubeyy/driver-churn-prediction-mlops.git
     cd driver-churn-prediction-mlops