import argparse

parser = argparse.ArgumentParser(description='Body Parts Segmentation')

#########################
# General Train Settings
#########################
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
parser.add_argument('--epoch', type=int, default=100, help='epoch (default: 100)')
parser.add_argument('--num_gpu', type=int, default=2, help='number of GPUs')
parser.add_argument('--device_idx', type=str, default='0,1', help='Gpu idx')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization lambda (default: 0)')
parser.add_argument('--decay_step', type=int, default=15, help='num epochs for decaying learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay gamma')
parser.add_argument('--log_period_as_iter', type=int, default=12800, help='num iter')
parser.add_argument('--validation_period_as_iter', type=int, default=3000, help='validation period in iterations')
parser.add_argument('--test', type=bool, default=False, help='is Test')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--backed_model', type=str, default='resnet50', help='Model: resnet50 | resnet101 | resnet152')
parser.add_argument('--seg_model', type=str, default='fcn', help='Seg Model (default: fcn)')
parser.add_argument('--random_seed', type=int, default=0, help='Random seed value')

#########################
# Experiment Logging Settings
#########################
parser.add_argument('--experiment_dir', type=str, default='', help='Experiment save directory')
parser.add_argument('--experiment_description', type=str, default='test', help='Experiment description')
parser.add_argument('--checkpoint_path', type=str, default='../checkpoints/best_acc_model.pth',
                    help='Checkpoint path')

args = parser.parse_args()