from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard
from pathlib import Path
from enum import StrEnum

class CLIArguments(StrEnum):
    yaml_file_location = "YAML Config File Location"
    experiment_name = "MLFlow Experiment name"
    run_name = "Run Name"
    logging_folder = "Logging Folder"

@dataclass
class HyperParameters:
    learning_rate: float
    epochs: int
    batch_size: int
    weight_decay: float

@dataclass
class TrainingConfig:
    train_split: float
    feature_columns: list[str]
    target_column: str

@dataclass
class YAMLconfig(YAMLWizard):
    data_file_name: str
    ml_flow_endpoint: str
    hyper_parameters: HyperParameters
    training_details: TrainingConfig



@dataclass
class CLIFields:
    yaml_file_location : Path
    experiment_name : str
    run_name : str
    log_folder : Path
    logging_file_name : Path = field(init=False)
    
    def __post_init__(self):
        if not self.yaml_file_location.exists():
            FileNotFoundError(f"File {self.yaml_file_location} not found.")
        
        if not self.log_folder.exists():
            FileNotFoundError(f'Direction {self.log_folder} not found.')
        
        self.logging_file_name = self.log_folder / self.experiment_name / self.run_name / '.log'