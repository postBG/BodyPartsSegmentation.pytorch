import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from metrics import eval_seg_metrics
from loggers import LoggerService
from misc import AverageMeterSet


class Trainer(object):
    def __init__(self, model, dataloaders, optimizer, criterion, num_epochs, args, num_classes,
                 log_period_as_iter=40000, train_loggers=None, val_loggers=None, device='cuda', train_evaluators=None,
                 val_evaluators=None, lr_scheduler=None, **kwargs):
        self.model = model.to(device)
        self.model = nn.DataParallel(self.model) if args.num_gpu > 1 else self.model

        self.args = args

        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.device = device

        self.logger_service = LoggerService(train_loggers, val_loggers)
        self.train_evaluators = train_evaluators if train_evaluators else []
        self.val_evaluators = val_evaluators if val_evaluators else []

        self.log_period_as_iter = log_period_as_iter
        self.validation_period_as_iter = args.validation_period_as_iter

        self.num_classes = num_classes

    def train(self):
        accum_iter = 0
        self.validate(0, self.dataloaders['val'], accum_iter)
        for epoch in range(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    _, accum_iter = self.train_one_epoch(epoch, self.dataloaders[phase], accum_iter)
                else:
                    self.validate(epoch, self.dataloaders['val'], accum_iter)

        self.logger_service.complete({
            'state_dict': (self._create_state_dict())
        })

    def train_one_epoch(self, epoch, dataloader, accum_iter):
        self.model.train()

        self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(dataloader, ncols=150)

        for batch_idx, (inputs, gt_mask) in enumerate(tqdm_dataloader):
            batch_size = inputs.size(0)
            inputs, gt_mask = inputs.to(self.device), gt_mask.type(torch.LongTensor).to(self.device)

            self.optimizer.zero_grad()

            # Source CE Loss

            logits = self.model(inputs)
            ce_loss = self.criterion(logits, gt_mask)
            ce_loss.backward()

            self.optimizer.step()

            average_meter_set.update('ce_loss', ce_loss.item())
            tqdm_dataloader.set_description(
                "Epoch {}, ce_loss {:.3f}, ".format(epoch + 1, average_meter_set['ce_loss'].avg))

            accum_iter += batch_size

            if self._is_logging_needed(accum_iter):
                tqdm_dataloader.set_description("Logging to Tensorboard...")
                log_data = {
                    'state_dict': (self._create_state_dict()),
                    'epoch': epoch,
                    'accum_iter': accum_iter
                }
                log_data.update(average_meter_set.averages())
                self.logger_service.log_train(log_data)

            if self._is_validation_needed(accum_iter):
                self.validate(epoch, self.dataloaders['val'], accum_iter)
                self.model.train()
        return accum_iter

    def _is_logging_needed(self, accum_iter):
        return accum_iter % self.log_period_as_iter < self.args.batch_size and accum_iter != 0

    def _is_validation_needed(self, accum_iter):
        return accum_iter % self.validation_period_as_iter < self.args.batch_size and accum_iter != 0

    def validate(self, epoch, dataloader, accum_iter):
        gts_all, predictions_all = [], []

        self.model.eval()

        with torch.no_grad():
            tqdm_dataloader = tqdm(dataloader)
            for batch_idx, (target_inputs, target_gt_mask) in enumerate(tqdm_dataloader):
                target_inputs, target_gt_mask = target_inputs.to(self.device), target_gt_mask.numpy()

                target_logits = self.model(target_inputs)
                _, target_predictions = target_logits.max(1)
                target_predictions = target_predictions.cpu().numpy()

                predictions_all.append(target_predictions)
                gts_all.append(target_gt_mask)

            gts_all = np.concatenate(gts_all)
            predictions_all = np.concatenate(predictions_all)

            acc, acc_cls, mean_iou, fwavacc, iou = eval_seg_metrics(predictions_all, gts_all, self.num_classes)
            print("Acc: {:.3f}, mIOU: {:.3f}".format(acc, mean_iou))

            log_data = {
                'state_dict': (self._create_state_dict()),
                'epoch': epoch,
                'accum_iter': accum_iter,
                'acc': acc,
                'mean_iou': mean_iou,
            }
            self.logger_service.log_val(log_data)

    def _create_state_dict(self):
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
