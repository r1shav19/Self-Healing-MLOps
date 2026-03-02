import subprocess


def retrain_model():
    print("🚀 Drift detected — starting retraining...")

    subprocess.run(
        ["python", "-m", "pipeline.training_pipeline"],
        check=True
    )

    print("✅ New model trained successfully!")