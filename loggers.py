import os
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torchvision.transforms import ToTensor

from utils import colorize_mask


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class LoggerService(object):
    def __init__(self, train_loggers=None, val_loggers=None):
        self.train_loggers = train_loggers if train_loggers else []
        self.val_loggers = val_loggers if val_loggers else []

    def complete(self, log_data):
        for logger in self.train_loggers:
            logger.complete(**log_data)
        for logger in self.val_loggers:
            logger.complete(**log_data)

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, metric_key='mIOU', filename='best_acc_model.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        current_metric = kwargs[self.metric_key]
        if self.best_metric < current_metric:
            print("Update Best {} Model at {}".format(self.metric_key, kwargs['epoch']))
            self.best_metric = current_metric
            save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename)


class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_label='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_label
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        if self.key in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        else:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])

    def complete(self, *args, **kwargs):
        self.writer.close()


class ImagePrinter(AbstractBaseLogger):
    """
    Input PIL images directly sampled from dataset classes to gt_images and input_images
    """

    def __init__(self, writer, gt_images, input_images, model_key='model', is_train=True):
        self.model_key = model_key
        self.writer = writer
        self.is_train = is_train
        self.gt_images = gt_images
        self.input_images = input_images
        self.mode = 'train' if is_train else 'val'
        self.to_tensor = ToTensor()

        for i, img in enumerate(self.input_images):
            self.writer.add_image('images_{}/{}/Input Image'.format(self.mode, i), img, 0)

        for i, img in enumerate(self.gt_images):
            self.writer.add_image('images_{}/{}/GT Mask'.format(self.mode, i),
                                  self.to_tensor(colorize_mask(img).convert('RGB')), 0)

    def log(self, *args, **kwargs):
        model = kwargs[self.model_key]

        for i, img in enumerate(self.input_images):
            prediction = self.evaluate(img, model)
            self.writer.add_image('images_{}/{}/Prediction'.format(self.mode, i),
                                  self.to_tensor(colorize_mask(np.squeeze(prediction, 0)).convert('RGB')),
                                  kwargs['accum_iter'])

    @staticmethod
    def evaluate(img, model):
        """
        Returns tensor of output prediction
        """
        is_train = model.training
        with torch.no_grad():
            img = img.to('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            output = model(img.unsqueeze(0))
            prediction = output.data.max(1)[1].cpu().numpy()
            if is_train:
                model.train()
        return prediction

    def complete(self, *args, **kwargs):
        self.writer.close()
