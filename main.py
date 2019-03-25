import os
import pprint as pp

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from datasets import dataloaders_factory
from datasets.pascal_parts import CLASS_WEIGHT, IGNORE_LABEL
from loggers import MetricGraphPrinter, RecentModelLogger, BestModelLogger
from misc import create_experiment_export_folder, export_experiments_config_as_json, fix_random_seed_as, set_up_gpu
from models import model_factory
from options import args as parsed_args
from trainer import Trainer


def main(args):
    export_root, args = setup_experiments(args)
    device = args.device
    model_checkpoint_path = os.path.join(export_root, 'models')

    dataloaders = dataloaders_factory(args)
    model = model_factory(args)

    writer = SummaryWriter(os.path.join(export_root, 'logs'))

    train_loggers = [
        MetricGraphPrinter(writer, key='ce_loss', graph_name='ce_loss', group_name='Train'),
        MetricGraphPrinter(writer, key='epoch', graph_name='Epoch', group_name='Train')
    ]
    val_loggers = [
        MetricGraphPrinter(writer, key='mean_iou', graph_name='mIOU', group_name='Validation'),
        MetricGraphPrinter(writer, key='acc', graph_name='Accuracy', group_name='Validation'),
        RecentModelLogger(model_checkpoint_path),
        BestModelLogger(model_checkpoint_path, metric_key='mean_iou'),
    ]

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, weight=torch.Tensor(CLASS_WEIGHT).to(device))
    optimizer = create_optimizer(model, args)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)

    trainer = Trainer(model, dataloaders, optimizer, criterion, args.epoch, args, num_classes=42,
                      log_period_as_iter=args.log_period_as_iter, train_loggers=train_loggers,
                      val_loggers=val_loggers, lr_schedulers=scheduler)
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
