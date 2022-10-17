## Core Engine for object detection task

### Structure
#### args.py
check `args.py` for detailed arg usage

To be continue...

#### config.py
#### engine/train.py
#### engine/infer.py
#### engine/eval.py


#### Experiment File Storage

- Experiment name: config.exp_name
```
expname_cfg.yaml
expname_args.yaml
expname.log
expname_loss.log
best_epoch.pth or last_epoch.pth
```

- Resume Rule:
 1. modified formal loss.log if its last epoch record is incomplete
 2. read the last_epoch.pth as checkpoint
### About Input Model
The input Model must have the following function:
```
Model.set(args, device)
    -args: args from get_train_args_parser()
    -device: device
```
```
Model.get_stats()
    Used after Infer and Eval.
    No input value are used.
    Simply define a void func if you don't want to use it.
```
```
Model.coco_parse_result(List: list[result])
    -Input: List: list[result]
    -Output: List: list[{coco_pred1},{coco_pred2}...]
```
*Copyright© NuayHL 2022. All right reserved*