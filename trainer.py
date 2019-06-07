import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from metrics import eval_seg_metrics
from loggers import LoggerService
from misc import AverageMeterSet
from utils import save_images_for_debugging
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels

OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'

STATE_DICT_KEY = 'model_state_dict'


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
        self.debug = args.debug

        self.num_classes = num_classes

    def train(self):
        accum_iter = 0

        # self.validate(0, self.dataloaders['val'], accum_iter)
        for epoch in range(self.num_epochs):
            # for phase in ['train', 'val']:
            for phase in ['train', 'val']:
                if phase == 'train':
                    accum_iter = self.train_one_epoch(epoch, self.dataloaders[phase], accum_iter)
                else:
                    self.validate(epoch, self.dataloaders['val'], accum_iter)

        self.logger_service.complete({
            'state_dict': (self._create_state_dict())
        })

    def train_one_epoch(self, epoch, dataloader, accum_iter):
        self.model.train()

        self.model.backbone.requires_grad = False

        self.lr_scheduler.step()

        average_meter_set = AverageMeterSet()
        tqdm_dataloader = tqdm(dataloader, ncols=150)

        for batch_idx, (inputs, gt_mask) in enumerate(tqdm_dataloader):
            batch_size = inputs.size(0)

            if batch_size <= 1:
                continue

            inputs, gt_mask = inputs.to(self.device), gt_mask.type(torch.LongTensor).to(self.device)

            if self.debug:
                save_images_for_debugging(batch_idx, gt_mask, inputs)

            self.optimizer.zero_grad()

            # Source CE Loss
            logits = self.model(inputs)
            loss = self.criterion(logits, gt_mask)
            loss.backward()

            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            tqdm_dataloader.set_description(
                "Epoch {}, loss {:.8f}, ".format(epoch + 1, average_meter_set['loss'].avg))

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

                if self.args.crf:
                    target_predictions = []
                    target_logits = target_logits.data.cpu().numpy()
                    for i in target_logits:
                        d = dcrf.DenseCRF2D(513, 513, 25)
                        U = unary_from_softmax(i)
                        d.setUnaryEnergy(U)
                        d.addPairwiseGaussian(sxy=3, compat=3)
                        Q = d.inference(5)
                        after_crf = np.argmax(Q, axis=0).reshape((513, 513))
                        target_predictions.append(after_crf)

                    target_predictions = np.array(target_predictions)
                else:
                    _, target_predictions = target_logits.max(1)
                    target_predictions = target_predictions.cpu().numpy()

                predictions_all.append(target_predictions)
                gts_all.append(target_gt_mask)

            gts_all = np.concatenate(gts_all)
            predictions_all = np.concatenate(predictions_all)

            acc, acc_cls, mean_iou, fwavacc, iou = eval_seg_metrics(predictions_all, gts_all, self.num_classes)
            print("Acc: {:.3f}, mIOU: {:.3f}".format(acc, mean_iou))

            log_data = {
                'state_dict': self._create_state_dict(),
                'epoch': epoch,
                'accum_iter': accum_iter,
                'acc': acc,
                'mean_iou': mean_iou,
                'model': self.model,
            }
            with open('merged.txt', 'a') as f:
                print('epoch:', epoch, 'iou: ', iou, 'miou: ', mean_iou, file=f)

            self.logger_service.log_val(log_data)

    def _create_state_dict(self):
        return {
            STATE_DICT_KEY: self.model.state_dict(),
            OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict(),
        }
