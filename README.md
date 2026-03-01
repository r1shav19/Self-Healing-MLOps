# Self-Healing MLOps Platform

A comprehensive machine learning operations platform with automated retraining, monitoring, and self-healing capabilities.

## Features

- **Automated Data Ingestion**: Streamlined data pipeline with validation
- **Preprocessing**: Data cleaning, transformation, and feature engineering
- **Model Training**: Scalable training with versioning and experiment tracking
- **Evaluation**: Comprehensive model evaluation and metrics tracking
- **API**: FastAPI-based REST API for model inference
- **Monitoring**: Real-time performance monitoring and alerting
- **Self-Healing**: Automated retraining when performance degrades
- **Configuration Management**: YAML-based configuration system

## Project Structure

```
self-healing-mlops/
├── data/              # Raw and processed data
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Core ML modules
├── pipeline/          # Training pipeline orchestration
├── models/            # Trained models
├── api/               # FastAPI inference server
├── monitoring/        # Monitoring and alerting
├── retraining/        # Automated retraining logic
├── config/            # Configuration files
├── logs/              # Application logs
└── requirements.txt   # Python dependencies
```

## Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd self-healing-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Usage

### Data Ingestion
```bash
python src/data_ingestion.py --config config/data.yaml
```

### Training
```bash
python -m pipeline.training_pipeline --config config/training.yaml
```

### API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Monitoring
```bash
python monitoring/monitor.py --config config/monitoring.yaml
```

## Configuration

All configuration is stored in `config/` directory as YAML files:

- `data.yaml` - Data ingestion settings
- `training.yaml` - Model training parameters
- `api.yaml` - API configuration
- `monitoring.yaml` - Monitoring and alerting rules

## Development

### Running Tests
```bash
pytest tests/ -v --cov=src
```

### Code Quality
```bash
# Format code
black src/ pipeline/ api/ monitoring/ retraining/

# Lint
flake8 src/ pipeline/ api/ monitoring/ retraining/

# Type checking
mypy src/ pipeline/ api/ monitoring/ retraining/
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions, please create an issue on GitHub.
