import json
from enum import Enum
import time



class Logger:
    
    class LogLevel(Enum):
        DEBUG = 0
        INFO = 1
        WARNING = 2
        ERROR = 3

        def __str__(self):
            return self.name

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, print_log_level=LogLevel.INFO):
        if not hasattr(self, "_initialized"):
            self.print_log_level = print_log_level
            self._initialized = True

    def initialize_wandb(self, project_name):
        import wandb
        wandb.init(project=project_name)
        self._wandb_available = True    
    
    def print_config(self, args):
        if getattr(self, "_wandb_available", False):
            wandb.config.update(args)
        print("Configuration:")
        print(json.dumps(args, indent=4))
        

    def log(self, log_data: dict, log_level: LogLevel = LogLevel.INFO):
        """
        Logs the provided information.

        Parameters:
        - log_data (dict): A dictionary containing the information to log.
        - log_level (LogLevel): The log level for the message. Default is LogLevel.INFO.

        Behavior:
        - Logs the information to the console and, if enabled, to the wandb service.
        - The log level determines if the log is printed in the logs.
        """            
                    
        if getattr(self, "_wandb_available", False):
            wandb.log(log_data)
        
        if log_level.value >= self.print_log_level.value:
            print(f"[{log_level}] {time.strftime('%Y-%m-%d %H:%M:%S')}: ", end="")
            print(json.dumps(log_data, indent=4))
            
        
