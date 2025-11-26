from typing import Protocol
from dataclasses import dataclass, field
from pathlib import Path
from argparse import ArgumentParser
from enum import StrEnum

class CLIArguments(StrEnum):
    yaml_file_location = "YAML Config File Location"
    experiment_name = "MLFlow Experiment name"
    run_name = "Run Name"
    logging_folder = "Logging Folder"



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

class CLI(Protocol):
    def get_cli_arguments(self)->CLIFields:...

class ArgParseCLI:
    def __init__(self) -> None:
        self.parser = self._setup_cli()
    
    def _setup_cli(self)->ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(CLIArguments.yaml_file_location.value)
        parser.add_argument(CLIArguments.experiment_name.value)
        parser.add_argument(CLIArguments.run_name.value)
        parser.add_argument(CLIArguments.logging_folder.value)
        return parser
    
    def get_cli_arguments(self)->CLIFields:
        field_values = vars(self.parser.parse_args())
        
        yaml_location_path = Path(field_values[CLIArguments.yaml_file_location.value])
        experiment_name = field_values[CLIArguments.experiment_name.value]
        run_name = field_values[CLIArguments.run_name.value]
        logging_folder = Path(field_values[CLIArguments.logging_folder.value])
        
        
        return CLIFields(yaml_file_location=yaml_location_path,
                         experiment_name=experiment_name,
                         run_name=run_name,
                         log_folder=logging_folder)