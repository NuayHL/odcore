import argparse

def get_train_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Default odcore args', add_help=add_help)
    parser.add_argument('--resume-exp', default='', type=str, help='resume training from exp path')
    parser.add_argument('--conf-file', default='', type=str, help='config file path')
    parser.add_argument('--ckpt-file', default='', type=str, help='load the ckpt file which is compatible with config ')
    parser.add_argument('--fine-tune', default='', type=str, help='load the ckpt file only for model loading')
    parser.add_argument('--batch-size', default=32, type=int, help='number of batchsize')
    parser.add_argument('--accumu', default=1, type=int, help='loss accumulate times. Set for larger virtual batchsize')
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=20, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--save-interval', default=50, type=int, help='save ckpt at every interval epochs')
    parser.add_argument('--step-lr', action='store_true', help='if using step wise learning rate')
    parser.add_argument('--safety-mode', action='store_true', help='if using loss protect measures')
    return parser

def get_infer_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Default infer odcore args', add_help=add_help)
    parser.add_argument('--img', default='', type=str, help='Img file path')
    parser.add_argument('--video', default='', type=str, help='Video file path')
    parser.add_argument('--conf-file', default='', type=str, help='config file path')
    parser.add_argument('--ckpt-file', default='', type=str, help='load the .pth file which is compatible with config ')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--size', default='ori', type=str, help='output size, i.e. hd, fhd, qhd, uhd or ori')
    parser.add_argument('--threshold', default=0.01, type=float, help='The score threshold for inference')
    parser.add_argument('--labeled', action='store_true', help='show bbox with label and score')
    parser.add_argument('--forward-func', default='', type=str, help='using other forward func')
    return parser

def get_eval_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Default infer odcore args', add_help=add_help)
    parser.add_argument('--conf-file', default='', type=str, help='config file path')
    parser.add_argument('--ckpt-file', default='', type=str, help='load the .pth file which is compatible with config ')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', default=8, type=int, help='workers for the loader')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size for the loader')
    parser.add_argument('--type', default='coco', type=str, help='choose evaluation metrics in [mip, coco, mr]')
    parser.add_argument('--force-eval', action='store_true', help='ignore the previous prediction cache and predict again')
    parser.add_argument('--forward-func', default='', type=str, help='using other forward func')
    return parser

if __name__ == '__main__':
    args = get_train_args_parser().parse_args()
