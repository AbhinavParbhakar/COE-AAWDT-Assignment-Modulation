from .providers import ArgParseCLI
import logging
from logging import Logger
from pathlib import Path
import mlflow
from logging import Logger


class Application:
    def __init__(self) -> None:
        self._cli = ArgParseCLI()
        self._logger : None | Logger = None
        
    def setup_logger(self, file_name:Path)->Logger:
        logger = logging.getLogger(__name__)
        logging.basicConfig(filename=file_name,filemode='w',datefmt='%m-%d-%Y %H-%M',level=logging.INFO)
        
        return logger

    

if __name__ == "__main__":
    cli = ArgParseCLI()
    cli_fields = cli.get_cli_arguments()
    logger = setup_logger(cli_fields.logging_file_name)
    