import os

# Define the folder structure
structure = {
    "industrial-motor-health": {
        "data": {
            "raw": ["motor_fault_dataset.csv"],
            "processed": ["processed_motor_data.parquet"],
            "sample": ["sample_input.json"]
        },
        "notebooks": [
            "01_data_exploration.ipynb",
            "02_feature_engineering.ipynb",
            "03_model_training.ipynb",
            "04_model_evaluation.ipynb"
        ],
        "src": {
            "data": ["ingest_data.py", "preprocess_data.py", "feature_engineering.py"],
            "models": ["train_model.py", "predict_model.py", "evaluate_model.py"],
            "pipelines": ["training_pipeline.py", "inference_pipeline.py"],
            "utils": ["config.py", "logger.py"]
        },
        "deployment": {
            "azure_function_app": ["predict_motor_fault.py"],
            "docker": ["Dockerfile"]
        },
        "mlflow": ["tracking_config.yaml"],
        "tests": ["test_data_processing.py", "test_model.py"],
        "configs": ["training_config.yaml"],
        "root_files": ["README.md", "requirements.txt", "setup.py", ".gitignore"]
    }
}

def create_structure(base, structure):
    for folder, content in structure.items():
        folder_path = os.path.join(base, folder)
        os.makedirs(folder_path, exist_ok=True)

        if isinstance(content, dict):
            create_structure(folder_path, content)
        elif isinstance(content, list):
            for file in content:
                file_path = os.path.join(folder_path, file)
                open(file_path, 'a').close()

# Create root folder and files
root = "industrial-motor-health"
os.makedirs(root, exist_ok=True)

# Create root-level files
for file in structure[root]["root_files"]:
    open(os.path.join(root, file), 'a').close()

# Create subfolders and files
for key, value in structure[root].items():
    if key != "root_files":
        create_structure(root, {key: value})

print("✅ Folder structure created successfully!")