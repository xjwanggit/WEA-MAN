import argparse

def args_parser():
    parser = argparse.ArgumentParser("training the model")
    ## dataset info
    parser.add_argument('-p', '--pretrained', default="", help='the path to pretrained model')
    parser.add_argument('--attack_type', default="", help='the method of the model stealing, e.g.logit_query, label_query')
    parser.add_argument('--arch', default='resnet50', help="the architecture of the encryption models")
    parser.add_argument('--data', default='/data/Newdisk/chenjingwen/code/First/data/Military_res', help='path to dataset')

    ## training info
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='numer of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='Weight decay (default: 1e-4)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true', help='use pin memory')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoitn, (default: None)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model_weights on validation set')
    parser.add_argument('--weight_path', type=str, default="/data/Newdisk/chenjingwen/code/First/save_model/attack/resnet50_Military_Model_water_CTW_quant.pth", help="the path of the model_weights weights")


    ## decrypt the model
    parser.add_argument('-d', '--decrypt_model', action='store_true', help='whether to decrypt the model or not')
    parser.add_argument('--license', type=str, default='/data/Newdisk/chenjingwen/biao3/Embedding_Code/license', help='the path of the license')


    parser.add_argument('--print_freq', '-f', default=1, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save_result', '-s', action='store_true', help='whether to save the output reusult')

    ## store file
    parser.add_argument('--save_path', default='/data/Newdisk/chenjingwen/code/First/save_model', help="the path to save training models")
    args = parser.parse_args()
    return args
