# SelfSupervised


## Config files
Config files which are in YAML format, include all hyperparameters and training parameters needed to train a model. Since each method might have different set of hyperparameters, the structure the file can be different among different methods.<br>


## Running an experiment

Under `scripts/` folder you can find the config and shell files to run each experiment (either training or evaluation). For example to train SimCLR on COCO2014 dataset, you need to run the corresponding shell from the root directory as below:
```
$ sh scripts/methods/simclr/coco2014/train.sh
```

## Running a down-stream task
In each `script` folder under `scripts/` there is a separate shell file for each down-stream task. As the parameters for running each tasks is variable and is dependent on the task, the arguments are parsed dynamically in the downstream runner, therefore it can  accept any argument in the form of `--<KEY> <VALUE>` as input.

# Highlighted results
A quick view of highlighted results.<br>

Please refer to the README page of each method for more detailed desciptions and results.

### SimCLR: Feature visualization on COCO2014's validation set 

<img src="assets/highlights/simclr_coco2014.png" width=400>

