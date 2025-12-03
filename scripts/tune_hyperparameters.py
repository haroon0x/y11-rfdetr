from ultralytics import YOLO

def tune_hyperparameters():
    # Load a model
    model = YOLO("yolo11m.pt")

    # Start tuning
    # This runs a genetic algorithm to find the best hyperparameters
    # It requires significant compute time
    print("Starting Hyperparameter Evolution...")
    print("This process will train the model multiple times with different settings.")
    
    model.tune(
        data="data/visdrone.yaml", # Start with VisDrone
        epochs=30,                 # Fewer epochs per trial for speed
        iterations=50,             # Number of trials
        optimizer="AdamW",         # Good default for transformers/modern CNNs
        plots=False,
        save=False,
        val=True
    )

if __name__ == "__main__":
    tune_hyperparameters()
