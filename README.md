## Core Engine for object detection task

#### Experiment File Storage

Experiment name: config.exp_name
```
expname_cfg.yaml
expname_args.yaml
expname.log
expname_loss.log
best_epoch.pth or last_epoch.pth
```

Resume Rule:
- modified formal loss.log if its last epoch record is incomplete
- read the last_epoch.pth as checkpoint
#### About Model
The input Model must have two member function:
```
Model.set(args, device)
# args: args from get_train_args_parser()
# device: device
```
```
Model.coco_parse_result(List: list[result])

# Input: List: list[result]
# Output: List: list[{coco_pred1},{coco_pred2}...]
```
