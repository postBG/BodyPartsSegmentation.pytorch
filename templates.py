def set_template(args):
    print("Template setting: {}".format(args.template))

    if args.template == 'proximal_log_ce':
        args.test = True
        args.batch_size = 4
        args.lr = 0.0005
        args.epoch = 10
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.transform_type = 'none'
        args.experiment_description = 'with_proximal_log_weighted_ce'
        args.criterion = 'ce'
        args.class_weight = 'proximal_log'

    elif args.template == 'dice':
        args.test = False
        args.batch_size = 4
        args.lr = 0.007
        args.epoch = 120
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.transform_type = 'none'
        args.experiment_description = 'with_dice_loss'
        args.criterion = 'dice'

    else:
        raise ValueError("Pick a correct template.")
