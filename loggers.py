import os
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torchvision.transforms import ToTensor

from datasets import STATISTICS_SET
from utils import colorize_mask, create_dataset_for_visualization, tensor_to_PIL


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
    def __init__(self, checkpoint_path, metric_key='mean_iou', filename='best_acc_model.pth'):
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
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
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

    def __init__(self, writer, dataset, indices=None, model_key='model', log_prefix='train'):
        self.model_key = model_key
        self.writer = writer
        self.dataset_name = type(dataset).__name__
        self.dataset = create_dataset_for_visualization(dataset, indices)
        self.log_prefix = log_prefix
        self.to_tensor = ToTensor()

        for i, (img, mask) in enumerate(self.dataset):
            self.writer.add_image('{}_{}/{}/Input Image'.format(self.log_prefix, self.dataset_name, i),
                                  self.to_tensor(tensor_to_PIL(img, **STATISTICS_SET)), 0)
            self.writer.add_image('{}_{}/{}/GT Mask'.format(self.log_prefix, self.dataset_name, i),
                                  self.to_tensor(colorize_mask(mask).convert('RGB')), 0)

    def log(self, *args, **kwargs):
        model = kwargs[self.model_key]

        for i, (img, _) in enumerate(self.dataset):
            prediction = self.evaluate(img, model)
            self.writer.add_image('{}/{}/Prediction'.format(self.log_prefix, i),
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
            output, x, y, z = model(img.unsqueeze(0))
            prediction = output.data.max(1)[1].cpu().numpy()
            if is_train:
                model.train()
        return prediction

    def complete(self, *args, **kwargs):
        self.writer.close()
