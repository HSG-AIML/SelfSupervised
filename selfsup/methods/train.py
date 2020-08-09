from selfsup.utils.limit_threads import * 
import argparse
import os
import importlib
from selfsup.utils.generic import load_config

def main(args):
    # Load config
    config = load_config(os.path.join(args.config_path, "config.yml"))
    
    # Get method trainer dynamically
    method_trainer = importlib.import_module(f"selfsup.methods.{config['method']}.trainer")
    Trainer = getattr(method_trainer, "Trainer")
    trainer = Trainer(config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    main(args)