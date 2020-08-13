from selfsup.utils.limit_threads import * 
import argparse
import os
import importlib
from selfsup.utils.generic import load_config


def main(args):
    r"""Main function that defines and runs the trainer."""

    # Load yaml config file
    config = load_config(os.path.join(args.config_path, "config.yml"))
    
    # Get method trainer dynamically based on selected method
    method_trainer = importlib.import_module(f"selfsup.methods.{config['method']}.trainer")

    # Get trainer class corresponding to method
    Trainer = getattr(method_trainer, "Trainer")

    # Initialize trainer class
    trainer = Trainer(config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    # Init argument parser
    parser = argparse.ArgumentParser()

    # Add yaml config path
    parser.add_argument("--config_path", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()

    # Call main function
    main(args)