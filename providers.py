from typing import Protocol
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.colors import BASE_COLORS

from provider_types import CLIArguments, CLIFields


class CLI(Protocol):
    def get_cli_arguments(self)->CLIFields:...

class GraphGenerator(Protocol):
    def create_single_metric_comparison_graph(self,save_path: Path, shared_x_values:list, metric_name: str, series_values: list[list[float]], series_labels: list[str])->None:...

    def get_create_graphs(self)->list[Path]:...
    
class PltPlotter:
    def __init__(self) -> None:
        self._created_graphs : list[Path] = []
    
    def get_create_graphs(self)->list[Path]:
        return self._created_graphs
        
    def create_single_metric_comparison_graph(self,save_path: Path, shared_x_values:list, metric_name: str, series_values: list[list[float]], series_labels: list[str])->None:
        if len(series_labels) < 1:
            raise ValueError("At least one series must be provided")
        
        if len(series_values) != len(series_labels):
            raise ValueError(f"Length of {len(series_values)} for series_values does not match length {len(series_labels)} of series_labels.")
        
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title(f'{metric_name} over epochs')
        ax.set_ylabel(metric_name)
        ax.set_xlabel('Epochs')
        base_color_names = list(BASE_COLORS.keys())
        number_of_base_colors = len(base_color_names)
        
        prev_size = len(series_values[0])
        
        for i,series in enumerate(series_values):
            if len(series) != prev_size:
                raise ValueError(f'All lists provided in the series_values argument must be of equal size. Found heterogeneity.')

            ax.plot(shared_x_values,series, base_color_names[i % number_of_base_colors], label=series_labels[i])
        
        ax.legend()
        fig.savefig(save_path)
        self._created_graphs.append(save_path)

class ArgParseCLI:
    def __init__(self) -> None:
        self._parser = self._setup_cli()
    
    def _setup_cli(self)->ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(CLIArguments.yaml_file_location.value)
        parser.add_argument(CLIArguments.experiment_name.value)
        parser.add_argument(CLIArguments.run_name.value)
        parser.add_argument(CLIArguments.logging_folder.value)
        return parser
    
    def get_cli_arguments(self)->CLIFields:
        field_values = vars(self._parser.parse_args())
        
        yaml_location_path = Path(field_values[CLIArguments.yaml_file_location.value])
        experiment_name = field_values[CLIArguments.experiment_name.value]
        run_name = field_values[CLIArguments.run_name.value]
        logging_folder = Path(field_values[CLIArguments.logging_folder.value])
        
        
        return CLIFields(yaml_file_location=yaml_location_path,
                         experiment_name=experiment_name,
                         run_name=run_name,
                         log_folder=logging_folder)