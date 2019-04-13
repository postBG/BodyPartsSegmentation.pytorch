def set_template(args):
    print("Template setting: {}".format(args.template))

    if args.template == 'default':
        args.test=True
        args.batch_size = 4
        args.lr = 0.001
        args.epoch = 10
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.decay_step = 5
        args.gamma = 0.1
        args.weight_decay = 2e-5
        args.transform_type = 'none'
        args.experiment_description = 'default_run'

    else:
        raise ValueError("Pick a correct template.")
