# Entities are nothing but returned type of the configuration manager functions (methods). We are just making sure the configuration manager returns the variables in thr right tyoe. 
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration class for data ingestion.

    Attributes:
        root_dir (Path): Root directory for data ingestion.
        source_URL (str): URL of the data source.
        local_data_file (Path): Path to the local data file.
        unzip_dir (Path): Directory to unzip the data file.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration class for data validation.

    Attributes:
        root_dir (Path): Root directory for data validation.
        STATUS_FILE (str): File to store validation status.
        unzip_data_dir (Path): Directory of unzipped data.
        all_schema (dict): Dictionary containing schema information.
    """
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration class for data transformation.

    Attributes:
        root_dir (Path): Root directory for data transformation.
        data_path (Path): Path to the data file.
        preprocessor_obj_file_path (Path): Path to the preprocessor object file.
    """
    root_dir: Path
    data_path: Path
    preprocessor_obj_file_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration class for model training.

    Attributes:
        root_dir (Path): Root directory for model training.
        train_data_path (Path): Path to the training data file.
        test_data_path (Path): Path to the test data file.
        model_name (str): Name of the model.
        n_estimators (int): Number of estimators for the model.
        max_depth (int): Maximum depth of the trees in the model.
        learning_rate (int): Learning rate for the model.
        random_state (int): Random state for reproducibility.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        subsample (int): Subsample ratio of the training instance.
        min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        target_column (str): Name of the target column.
    """
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    n_estimators: int
    max_depth: int
    learning_rate: int
    random_state: int
    min_samples_split: int
    subsample: int
    min_samples_leaf: int
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Configuration class for model evaluation.

    Attributes:
        root_dir (Path): Root directory for model evaluation.
        test_data_path (Path): Path to the test data file.
        model_path (Path): Path to the model file.
        all_params (dict): Dictionary containing all parameters for the model.
        metric_file_name (Path): File name to store evaluation metrics.
        target_column (str): Name of the target column.
        mlflow_uri (str): URI for MLflow tracking.
    """
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
