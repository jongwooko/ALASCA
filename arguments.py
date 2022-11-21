import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Python Training')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                    help='type of dataset')
    parser.add_argument('--data_dir', type=str, default='/home/work/alasca/dataset')

    # Optimization options
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.02, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        help='weight decay coefficient')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum of SGD')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--loss-fn', default='ce', type=str, help='type of loss fn')
    parser.add_argument('--gamma', default=0.01, type=float, help='factor for learning rate decay')

    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--out', default='result',
                            help='Directory to output the result')

    # Miscs
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    # Device options
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    # Method options
    parser.add_argument('--r', type=float, default=0.8)
    parser.add_argument('--warm-up', type=int, default=10)
    parser.add_argument('--noise-type', type=str, default='sym', choices=['sym', 'asym', 'inst'])

    parser.add_argument('--sd', action='store_true')
    parser.add_argument('--temperature', default=3., type=float,
                            help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='weight of kd loss')
    parser.add_argument('--smooth', default=1.0, type=float,
                        help='label smoothing')
    
    # Method options for Multi-Networks
    parser.add_argument("--use_multi_networks", action="store_true")
    parser.add_argument("--multi_networks_method", type=str, choices=["coteach", "decouple", "coteach+"])
    parser.add_argument("--num_gradual", type=int)
    
    # Method options 2
    parser.add_argument("--use_sd", action="store_true")
    parser.add_argument("--use_ls", action="store_true")
    parser.add_argument("--use_als", action="store_true")
    parser.add_argument("--lam", default=2.0, type=float,
                        help="The coefficient for self-distillation")
    parser.add_argument("--cyclic_train", action="store_true")

    # ELR options & SCE options
    parser.add_argument('--beta', default=0.7, type=float,
                        help='elr hyperparameter beta / sce hyperparameter beta')
    parser.add_argument('--lmbda', default=3., type=float,
                        help='elr hyperparameter lmbda / sce hyperparmeter alpha')
    
    parser.add_argument('--do_grad', action='store_true')
    parser.add_argument('--do_confcheck', action='store_true')
    parser.add_argument('--do_corrcheck', action='store_true')
    
    # AAAI
    parser.add_argument('--alasca', action='store_true')
    parser.add_argument('--alasca_plus', action='store_true')
    parser.add_argument('--w1', default=2.0, type=float,
                        help='lambda for alasca')
    parser.add_argument('--w2', default=0.7, type=float,
                        help='ema weight for alasca')
    parser.add_argument('--w3', default=0.333, type=float,
                        help='sharpen temperature for alasca')
    
    
    parser.add_argument('--byot', action='store_true')
    parser.add_argument('--byot_kd', action='store_true')
    parser.add_argument('--byot_fd', action='store_true')
    
    parser.add_argument('--warmup', default=20, type=int,
                        help='warmup epoch for alasca plus')
    
    parser.add_argument('--position_all', action='store_true')
    parser.add_argument('--linear_evaluation', action='store_true')
    
    
    return parser.parse_args()