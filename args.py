import argparse
import yaml

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Default odcore args', add_help=add_help)
    parser.add_argument('--conf-file', default='test_config.yaml', type=str, help='config file path')
    parser.add_argument('--ckpt-file', default='last_epoch.pth', type=str, help='load the ckpt file which is compatible with config ')
    parser.add_argument('--fine-tune', default='', type=str, help='load the ckpt file only for model loading')
    parser.add_argument('--batch-size', default=8, type=int, help='number of batchsize')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=20, type=int, help='evaluate at every interval epochs')
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    dict = vars(args)
