import os
import pprint as pp
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from datasets import dataloaders_factory, dataset_factory
from loggers import MetricGraphPrinter, RecentModelLogger, BestModelLogger, ImagePrinter
from losses import create_criterion
from misc import create_experiment_export_folder, export_experiments_config_as_json, fix_random_seed_as, set_up_gpu
from models import model_factory
from options import args as parsed_args
from trainer import Trainer


def main(args):
    export_root, args = setup_experiments(args)
    device = args.device
    model_checkpoint_path = os.path.join(export_root, 'models')

    train_dataset = dataset_factory(args.train_transform_type, is_train=True)
    val_dataset = dataset_factory(args.val_transform_type, is_train=False)

    dataloaders = dataloaders_factory(train_dataset, val_dataset, args.batch_size, args.test)
    model = model_factory(args)

    writer = SummaryWriter(os.path.join(export_root, 'logs'))

    train_loggers = [
        MetricGraphPrinter(writer, key='loss', graph_name='loss', group_name='Train'),
        MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train'),
    ]
    val_loggers = [
        MetricGraphPrinter(writer, key='mean_iou', graph_name='mIOU', group_name='Validation'),
        MetricGraphPrinter(writer, key='acc', graph_name='Accuracy', group_name='Validation'),
        RecentModelLogger(model_checkpoint_path),
        BestModelLogger(model_checkpoint_path, metric_key='mean_iou'),
        ImagePrinter(writer, train_dataset, log_prefix='train'),
        ImagePrinter(writer, val_dataset, log_prefix='val')
    ]

    criterion = create_criterion(args)
    optimizer = create_optimizer(model, args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

    if args.resume_training:
        exp_dir = os.path.join(args.experiment_dir, args.resume_training)
        model_dir = 'deeplab-resnet.pth.tar'
        chk = torch.load(model_dir)
        model_state = model.state_dict()
        pretrained_state = chk['state_dict']
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            k in model_state and v.size() == model_state[k].size()}

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
        # model.load_state_dict(chk['model_state_dict'])
        # optimizer.load_state_dict(chk['optimizer_state_dict'])
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)

    trainer = Trainer(model, dataloaders, optimizer, criterion, args.epoch, args, num_classes=args.classes,
                      log_period_as_iter=args.log_period_as_iter, train_loggers=train_loggers,
                      val_loggers=val_loggers, lr_scheduler=scheduler, device=device)
    trainer.train()
    writer.close()


def create_optimizer(model, args):
    if args.optimizer == 'Adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)


def setup_experiments(args):
    set_up_gpu(args)
    fix_random_seed_as(args.random_seed)

    export_root = create_experiment_export_folder(args)
    export_experiments_config_as_json(args, export_root)

    pp.pprint(vars(args), width=1)
    return export_root, args


if __name__ == "__main__":
    main(parsed_args)