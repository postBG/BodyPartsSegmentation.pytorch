def set_template(args):
    print("Template setting: {}".format(args.template))

    if args.template == 'proximal_log_ce':
        args.resume_training = 'with_proximal_log_weighted_ce_2019-04-25_2'
        args.test = False
        args.batch_size = 4
        args.lr = 0.001
        args.epoch = 120
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.train_transform_type = 'none'
        args.experiment_description = 'with_proximal_log_weighted_ce'
        args.criterion = 'ce'
        args.class_weight = 'proximal_log'

    elif args.template == 'resnet101_backbone':
        args.backbone = 'resnet'
        args.test = False
        args.batch_size = 4
        args.lr = 0.001
        args.epoch = 120
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.train_transform_type = 'none'
        args.experiment_description = 'with_proximal_log_weighted_ce'
        args.criterion = 'ce'
        args.class_weight = 'proximal_log'

    elif args.template == 'with_fcn_res50':
        args.test = False
        args.batch_size = 4
        args.lr = 0.001
        args.epoch = 120
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.train_transform_type = 'none'
        args.seg_model = 'fcn'
        args.experiment_description = 'with_fcn_res50_and_dice'
        args.criterion = 'ce'

    elif args.template == 'dice':
        # args.resume_training = 'deeplab-resnet.pth.tar'
        args.backbone = 'resnet'
        args.test = False
        args.batch_size = 4
        args.lr = 0.007
        args.epoch = 120
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.train_transform_type = 'random'
        args.experiment_description = 'with_pretrained_deeplab_dice'
        args.criterion = 'dice'
        args.classes = 25

    elif args.template == 'lovasz':
        args.pretrained_weights = '../seg_seungmin/experiments/lovasz_loss_adam_2019-05-24_0/models/best_acc_model.pth'  #05-18_1 with_pretrained_deeplab_2019-05-24_0
        args.backbone = 'resnet'
        args.test = True
        args.batch_size = 4
        args.lr = 1e-5
        args.epoch = 40
        args.optimizer = 'Adam'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.train_transform_type = 'random'
        args.experiment_description = 'lovasz_loss_adam'
        args.criterion = 'lovasz'
        args.classes = 25

    elif args.template == 'lovasz_multi':
        args.resume_training = 'experiments/lovasz_loss_adam_2019-05-24_0'
        args.backbone = 'resnet'
        args.test = True
        args.batch_size = 4
        args.lr = 1e-5
        args.epoch = 40
        args.optimizer = 'Adam'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.train_transform_type = 'random'
        args.experiment_description = 'lovasz_loss_adam_multi_scale_pred'
        args.criterion = 'lovasz'
        args.classes = 25

    elif args.template == 'merged':
        args.resume_training = 'with_merged_annotations_2019-05-18_0'
        args.backbone = 'resnet'
        args.test = False
        args.batch_size = 4
        args.lr = 0.001
        args.epoch = 120
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.train_transform_type = 'random'
        args.experiment_description = 'with_merged_annotations_after'
        args.criterion = 'ce'
        args.classes = 25
        args.class_weight = 'proximal_log'

    elif args.template == 'pretrained':
        args.pretrained_weights = 'deeplab-resnet.pth.tar'
        args.backbone = 'resnet'
        args.test = False
        args.batch_size = 4
        args.lr = 0.00001
        args.epoch = 30
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.train_transform_type = 'random'
        args.experiment_description = 'with_pretrained_deeplab'
        args.criterion = 'ce'
        args.classes = 25
        args.class_weight = 'naive'
    else:
        raise ValueError("Pick a correct template.")
