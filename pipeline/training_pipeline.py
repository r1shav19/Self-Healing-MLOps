from src.data_ingestion import load_data
from src.preprocessing import preprocess_data
from src.train import train_model


def run_training_pipeline():

    print("Loading fraud dataset...")
    df = load_data("data/raw/creditcard.csv")

    print("Preprocessing transactions...")
    X, y = preprocess_data(df)

    print("Training fraud detection model...")
    train_model(X, y)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()

def run_pipeline():
    print("Loading fraud dataset...")
    print("Preprocessing transactions...")
    print("Training fraud detection model...")
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()