import argparse

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Default odcore args', add_help=add_help)
    parser.add_argument('--conf-file', default='./configs/yolov6s.py', type=str, help='config file path')
    parser.add_argument('--ckpt-file', default='', type=str, help='load the ckpt file which is compatible with config ')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=20, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--eval-final-only', action='store_true', help='only evaluate at the final epoch')
    parser.add_argument('--name', default='exp', type=str, help='experiment name, saved to output_dir/name')
    return parser
