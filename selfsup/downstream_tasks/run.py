from selfsup.utils.limit_threads import * 
import sys
import os
import importlib
import torch
from selfsup.utils.generic import load_config


def get_cmd_params():
    r"""Retrieves list of parameters from command line and returns them as a dict."""
    args = sys.argv[1:]
    
    # Make sure number of arguments is even (must be key,value pair)
    assert len(args) % 2 ==0
    
    # Create CMD params
    cmd_params = {}
    for i in range(1,len(args), 2):
        # Remove -- from the beginning of keys
        key_name = args[i-1][2:]
        value = args[i]
        cmd_params[key_name] = value

    return cmd_params


def main(args):
    r"""Main function that runs a down-stream task."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load yaml config file
    config = load_config(os.path.join(args["config_path"], "config.yml"))
    
    # Get model module dynamically 
    model_module = importlib.import_module(f"selfsup.methods.{config['method']}.model")

    # Get model class corresponding to model_name
    model_module = getattr(model_module, config['model_name'])

    # Init model
    model = model_module(**config["model"]).to(device)
    checkpoint_path = args["checkpoint_path"]
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # Get task module dynamically 
    task_module = importlib.import_module(f"selfsup.downstream_tasks.{args['task_name']}.run_task")

    # Get execute function of the task
    execute = getattr(task_module, "execute")

    # Execute task
    args["input_shape"] = config["dataset"]["input_shape"]
    execute(args, model)
    

if __name__ == "__main__":
    # Get command-line params 
    args = get_cmd_params()

    # Call main function
    main(args)
