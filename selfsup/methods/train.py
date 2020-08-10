from selfsup.utils.limit_threads import * 
import argparse
import os
import importlib
from selfsup.utils.generic import load_config

# define main function
def main(args):

    # load yaml config file
    config = load_config(os.path.join(args.config_path, "config.yml"))
    
    # get method trainer dynamically based on selected method
    method_trainer = importlib.import_module(f"selfsup.methods.{config['method']}.trainer")

    # get trainer class corresponding to method
    Trainer = getattr(method_trainer, "Trainer")

    # initialize trainer class
    trainer = Trainer(config)

    # start training
    trainer.train()


if __name__ == "__main__":

    # init argument parser
    parser = argparse.ArgumentParser()

    # add yaml config path
    parser.add_argument("--config_path", type=str, required=True)

    # parse arguments
    args = parser.parse_args()

    # call main function
    main(args)