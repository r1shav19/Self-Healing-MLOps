from src.data_ingestion import load_data
from src.preprocessing import preprocess_data

def run_training_pipeline():

    print("Loading data...")
    data = load_data("data/raw/data.csv")

    print("Preprocessing...")
    data = preprocess_data(data)

    print("Pipeline setup complete.")


if __name__ == "__main__":
    run_training_pipeline()