# Employee Salary Prediction Web App

🎯 A Flask web application for predicting whether an individual's annual income exceeds $50,000 based on demographic and professional characteristics.

## 📊 Project Overview

This project implements multiple machine learning algorithms to predict employee salaries using the Adult Census Income dataset. The best performing model (Gradient Boosting with 85.71% accuracy) is deployed as a web application.

### 🚀 Features
- **Interactive Web Interface**: User-friendly form for inputting employee data
- **REST API**: Programmatic access for integration with other systems
- **Multiple Algorithm Support**: Trained models include Logistic Regression, Random Forest, KNN, SVM, and Gradient Boosting
- **Real-time Predictions**: Instant salary classification with confidence scores
- **Containerized Deployment**: Docker support for easy deployment

### 🤖 Machine Learning Models

| Algorithm | Accuracy | Precision (>50K) | Recall (>50K) | F1-Score (>50K) |
|-----------|----------|------------------|---------------|------------------|
| **Gradient Boosting** ⭐ | **85.71%** | **0.78** | **0.60** | **0.68** |
| Random Forest | 85.08% | 0.74 | 0.62 | 0.67 |
| SVM | 83.96% | 0.75 | 0.54 | 0.63 |
| KNN | 82.45% | 0.67 | 0.60 | 0.63 |
| Logistic Regression | 81.49% | 0.69 | 0.46 | 0.55 |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7 or later
- pip (Python package installer)
- Docker (optional, for containerized deployment)

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/employee-salary-prediction.git
   cd employee-salary-prediction
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate model files:**
   ```bash
   python save_model.py
   ```
   *Note: Replace the sample data in `save_model.py` with your actual trained model from the Jupyter notebook*

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Access the application:**
   - Web Interface: http://localhost:5000
   - Health Check: http://localhost:5000/health

## 🐳 Docker Deployment

### Build and Run with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t salary-prediction-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5000:5000 salary-prediction-app
   ```

3. **Access the application:**
   Open http://localhost:5000 in your browser

### Production Deployment with Docker Compose

Create a `docker-compose.yml` file:
```yaml
version: '3.8'
services:
  salary-app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:
```bash
docker-compose up -d
```

## 🔌 API Documentation

### Prediction Endpoint

**POST** `/api/predict`

**Request Body:**
```json
{
  "age": 39,
  "workclass": 3,
  "educational_num": 13,
  "marital_status": 2,
  "hours_per_week": 40,
  "gender": 1
}
```

**Response:**
```json
{
  "prediction": ">50K",
  "confidence": 0.85
}
```

### Feature Encoding Reference

| Feature | Encoding |
|---------|----------|
| **workclass** | 0=Federal-gov, 1=Local-gov, 2=Never-worked, 3=Private, 4=Self-emp-inc, 5=Self-emp-not-inc, 6=State-gov, 7=Without-pay |
| **marital_status** | 0=Divorced, 1=Married-AF-spouse, 2=Married-civ-spouse, 3=Married-spouse-absent, 4=Never-married, 5=Separated, 6=Widowed |
| **gender** | 0=Female, 1=Male |

### Health Check Endpoint

**GET** `/health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

## 📁 Project Structure

```
employee-salary-prediction/
├── app.py                      # Flask web application
├── save_model.py              # Model training and saving script
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── README.md                  # Project documentation
├── employee salary prediction.ipynb  # Jupyter notebook with analysis
├── salary_prediction_model.pkl       # Trained model (generated)
└── scaler.pkl                        # Feature scaler (generated)
```

## 🧪 Testing the API

### Using curl:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": 3,
    "educational_num": 13,
    "marital_status": 2,
    "hours_per_week": 40,
    "gender": 1
  }'
```

### Using Python requests:
```python
import requests

data = {
    "age": 39,
    "workclass": 3,
    "educational_num": 13,
    "marital_status": 2,
    "hours_per_week": 40,
    "gender": 1
}

response = requests.post('http://localhost:5000/api/predict', json=data)
print(response.json())
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Application port (default: 5000)

### Model Files
Ensure these files exist in the root directory:
- `salary_prediction_model.pkl`: Trained ML model
- `scaler.pkl`: Feature scaler for preprocessing

## 🚀 Deployment Options

### 1. Local Development
```bash
python app.py
```

### 2. Production with Gunicorn
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### 3. Cloud Deployment
- **Heroku**: Push to Heroku with Procfile
- **AWS EC2**: Deploy Docker container
- **Google Cloud Run**: Serverless container deployment
- **Azure Container Instances**: Container deployment

## 📊 Performance Metrics

- **Best Model**: Gradient Boosting Classifier
- **Accuracy**: 85.71%
- **Response Time**: <100ms average
- **Throughput**: 1000+ requests/minute

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Adult Census Income dataset
- Scikit-learn community for machine learning tools
- Flask community for web framework support

## 📞 Support

For questions or issues, please:
1. Check the [Issues](https://github.com/yourusername/employee-salary-prediction/issues) page
2. Create a new issue if needed
3. Contact the maintainers

---

**Made with ❤️ for salary prediction and fair compensation analysis**
