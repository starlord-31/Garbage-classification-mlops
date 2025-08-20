# Garbage Classification MLOps Project

## Project Overview

Proper classification of garbage is essential for efficient waste management and recycling. Traditionally, this process is manual, time-consuming, and error-prone due to the large volume and diversity of waste materials. Misclassification leads to contamination of recyclable materials, higher processing costs, and environmental harm.

Automating garbage classification with AI-powered image recognition can significantly enhance accuracy, speed, and scalability. However, building such an AI system requires careful coordination of data processing, model training, deployment infrastructure, monitoring, and continuous evaluation to ensure consistent performance in real-world conditions.

This project addresses these challenges by applying comprehensive MLOps practices to develop a robust garbage classification pipeline. This pipeline improves classification accuracy while providing production-ready deployment and monitoring solutions, enabling efficient and reliable waste sorting at scale.

**Note:** Large folders (`mlruns/`, `models/`, `prometheus-3.5.0.linux-amd64/`) are stored on Google Drive and must be downloaded separately.

---

## Data Source

The dataset used for training and evaluation in this project was sourced from Kaggle:  
[Garbage Dataset Classification](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification/data)

This publicly available dataset contains labeled images of various types of garbage, enabling the development of accurate classification models.

---

## 🏗️ Architecture Overview

The system implements a modular pipeline enabling reliable garbage classification:

```Data Ingestion → Data Preparation → Model Training → Model Deployment → Monitoring & Metrics → Continuous Evaluation```

---

## 📋 Workflow Components

### 1. Data Ingestion and Preparation

- Raw images are ingested and organized into training, validation, and test splits.
- Data loaders and preprocessing scripts standardize inputs for modeling.

### 2. Model Training and Experiment Tracking

- Multiple models trained (DenseNet, EfficientNet, MobileNet, ResNet, ViT).
- MLflow tracks experiments, metrics, and artifacts.
- Best models stored under `models/`.

### 3. Model Deployment

- FastAPI serves trained models through REST endpoints for inference.
- Uvicorn runs the ASGI server locally or in production environments.

### 4. Monitoring and Metrics

- Prometheus and Grafana monitor API usage and performance.
- Dashboards visualize latency, error rates, and throughput.

### 5. Continuous Evaluation and Testing

- Automated tests with `pytest`.
- CI/CD pipelines trigger linting, testing, and deployment.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker (optional)
- Node.js & npm (for frontend)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/starlord-31/Garbage-classification-mlops.git
cd Garbage-classification-mlops
```

2. **Create and activate a conda environment:**
```bash
conda create -n exp-tracking-env python=3.10 -y
conda activate exp-tracking-env
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download large folders from Google Drive:**
Download and place the following folders in the project root directory:

[Google Drive Folder Link](https://drive.google.com/drive/folders/1T7b_9VYo0Pt9AHX7ge5o-5isG4mxf2i_?usp=drive_link)

- `mlruns/`
- `models/`
- `prometheus-3.5.0.linux-amd64/`

5. **Install frontend dependencies:**
```bash
cd garbage-classifier-frontend
npm install
```

---

## 💻 Running the Application

### Train Models

### 1. Start Training Pipeline with Prefect
```bash
python src/prefect_pipeline.py
```
This orchestrates data loading, training, and experiment tracking.

### 2. Build and Run Docker Container

Build the Docker image:
```bash
docker build -t garbage-classifier:latest .
```
Run the container exposing port 8000:
```bash
docker run -p 8000:8000 garbage-classifier:latest
```

### 3. Run API Server Locally
```bash
uvicorn app:app --reload
```
API serves predictions at `http://localhost:8000`.

### 4. Run Frontend Application
```bash
cd garbage-classifier-frontend
npm install
npm start
```
Access the React UI at `http://localhost:3000`.

---

## 🔍 Monitoring with Prometheus and Grafana

### Prometheus

Prometheus binaries are included in the `prometheus-3.5.0.linux-amd64/` folder (downloaded from Google Drive).  
To start Prometheus:

```bash
cd prometheus-3.5.0.linux-amd64
./prometheus
```
This starts the Prometheus server which collects and stores metrics from the API and services.

---

### Grafana

Grafana dashboard is provided as JSON files in the `dashboards/` folder.

To visualize metrics using Grafana:

1. Install and run **Grafana** (see [Grafana installation guide](https://grafana.com/docs/grafana/latest/installation/)).

2. Open Grafana in the browser and log in.

3. Navigate to **“Manage” → “Import”**.

4. Upload the JSON dashboard files from the `dashboards/` folder.

5. Set up the Prometheus instance as the data source for Grafana.

This will provide rich visualization of metrics such as API latency, error rates, and throughput.

---

## ✔️ Testing

Run automated tests:
```bash
pytest tests/
```

---

## 🔄 Continuous Integration and Continuous Deployment (CI/CD)

This project uses GitHub Actions to automate CI/CD, ensuring code quality, testing, and Docker image building and publishing.

### Workflow Overview

- **Trigger:** Runs on pushes and pull requests to the `main` branch.
- **Steps:**  
  - Checkout code  
  - Set up Python 3.10 and cache dependencies  
  - Install Python packages (flake8, pytest)  
  - Lint code with flake8  
  - Run tests with pytest  
  - Build Docker image  
  - Log in to Docker Hub using GitHub secrets  
  - Push Docker image to Docker Hub  
  - Run frontend tests with Node.js and npm  

### Docker Hub Setup

To push Docker images, add your Docker Hub credentials as GitHub secrets under **Settings > Secrets and variables > Actions**:  
- `DOCKER_USERNAME` — your Docker Hub username  
- `DOCKER_PASSWORD` — your Docker Hub password or access token  

### How to Trigger

Push to `main` or open a pull request targeting `main` to automatically run the pipeline. Check progress and results in the **Actions** tab on GitHub.

This CI/CD pipeline helps automate testing, maintain quality, and manage container deployments efficiently.

---

## ⚙️ Development Workflow and Automation

### 1. Testing

- Added **integration tests** (`tests/test_integration.py`) using FastAPI’s TestClient to verify API endpoints including root, prediction, and invalid input handling.
- Implemented **API tests** (`tests/test_api.py`) that send real HTTP requests with image data for end-to-end verification.
- Tests are executed using `pytest` and require setting the Python path:
```bash
PYTHONPATH=. pytest tests/
```

### 2. Pre-commit Hooks

- Configured **pre-commit hooks** via `.pre-commit-config.yaml` to run code linting and formatting automatically before commits.
- Included hooks:
  - `flake8` for code style and lint checks.
  - Others like `isort` for import sorting if configured.
- Activate locally with:
```bash
pre-commit install
```

### 3. Makefile Automation

- Created a **Makefile** for common developer tasks:
  - `make lint` — runs linters and style checks.
  - `make test` — runs the full test suite with proper environment.
  - `make clean` — clears caches and temporary files.
- Simplifies running frequent commands consistently.

### 4. CI/CD Pipeline

- Pipeline automatically runs pre-commit checks and tests on commits and pull requests.
- Ensures code quality and reduces errors before merging.

---

### Code Style and Formatting

- Adjusted code for PEP 8 compliance:
  - Fixed line length issues.
  - Added required blank lines.
  - Removed unused imports.
- Recommended tools include **Black** and **isort** for formatting and import sorting.

---

## 📁 Project Structure

```
GARBAGE-CLASSIFICATION-MLOPS/
│
├── configs/
├── dashboards/
│   └── Grafana/
│   └── dashboard.json
├── data/
│   └── Garbage_Dataset_Classification/
│   ├── images/
│   └── metadata.csv
├── garbage-classifier-frontend/
│   ├── node_modules/
│   ├── public/
│   ├── src/
│   │   ├── App.css
│   │   ├── App.js
│   │   ├── App.test.js
│   │   ├── index.css
│   │   ├── index.js
│   │   ├── logo.svg
│   │   ├── reportWebVitals.js
│   │   └── setupTests.js
│   ├── .gitignore
│   ├── package-lock.json
│   ├── package.json
│   └── README.md
├── mlruns/
├── models/
├── node_modules/
├── prometheus-3.5.0.linux-amd64/
├── src/
│   ├── create_dataloaders.py
│   ├── data_preparation.py
│   ├── inference.py
│   ├── model.py
│   ├── prefect_pipeline.py
│   └── run_experiments.py
├── subset_data/
├── tests/
├── .github/
├── .pytest_cache/
├── .gitignore
├── app.py
├── Dockerfile
├── LICENSE
├── package-lock.json
├── package.json
├── README.md
└── requirements.txt
```

---

## 🔧 Key Components

### PrefectTrainingPipeline  
The core pipeline that orchestrates the end-to-end training workflow:  
- Data loading and preprocessing  
- Multi-model training and evaluation  
- Experiment tracking with MLflow  
- Model saving and versioning  

### FastAPI Serving  
Provides a REST API for inference:  
- Hosts trained models for garbage classification  
- Handles image input and returns predictions  
- Supports real-time interaction and batch requests  

### Prometheus & Grafana Monitoring  
Comprehensive monitoring of API performance and usage:  
- Collects metrics like request latency and error rates with Prometheus  
- Visualizes system health with customizable Grafana dashboards  

### MLflow Experiment Tracking  
Tracks and logs all training experiments:  
- Stores metrics, parameters, and artifacts  
- Enables reproducibility and model comparison  

---

## 📊 Performance Metrics

The project delivers strong metrics across key model and system indicators:  
- **High Accuracy (>85%)**: Reliable garbage type classification  
- **Robust Precision and Recall**: Balanced ability to correctly classify and discover relevant images  
- **Low Latency (<100ms)**: Fast inference response for API clients  
- **Stable System Monitoring**: Continuous visibility into API health and usage  

---

## 🎯 Use Cases

This project can be applied to:  
- Automated waste sorting and recycling systems  
- Environmental data collection and analysis  
- Smart city trash management solutions  
- Educational tools for computer vision and MLOps  
- Industrial IoT systems for waste monitoring  

---

## 🔮 Future Enhancements

Planned improvements include:  
- Real-time drift detection and model retraining  
- Container orchestration and scalable cloud deployment    
- Extended model architectures for improved accuracy  
- Integration with additional data monitoring and alerting tools  

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests to help improve the project.



