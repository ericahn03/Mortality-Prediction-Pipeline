import subprocess
import sys

# Define the pipeline steps and script paths
steps = [
    ("Preprocessing Data", "model_pipeline/preprocess_data.py"),
    ("Loading and Exploring Data", "model_pipeline/load_explore.py"),
    ("Training Base Models", "model_pipeline/train_models.py"),
    ("Training Ensemble Models", "model_pipeline/ensemble_models.py"),
    ("Visualizing Results", "model_pipeline/visualize_results.py"),
]

# Run each step and check for errors
for title, script in steps:
    print(f"\n--- {title} ---")
    result = subprocess.run(["python3", script])
    if result.returncode != 0:
        print(f"\n ERROR: {script} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

print("\n All steps completed successfully!")
