import subprocess

print("\n--- Loading and Exploring Data ---")
subprocess.run(["python3", "model_pipeline/load_explore.py"])

print("\n--- Preprocessing Data ---")
subprocess.run(["python3", "model_pipeline/preprocess_data.py"])

print("\n--- Training Base Models ---")
subprocess.run(["python3", "model_pipeline/train_models.py"])

print("\n--- Training Ensemble Models ---")
subprocess.run(["python3", "model_pipeline/ensemble_models.py"])

print("\n--- Visualizing Results ---")
subprocess.run(["python3", "model_pipeline/visualize_results.py"])

print("\nAll steps completed successfully!")