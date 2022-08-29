from deepscm.experiments import morphomnist  # noqa: F401
from .base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY

if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    import argparse
    import os
    import warnings

    
    exp_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_parser.add_argument('--experiment', '-e', help='which experiment to load', choices=tuple(EXPERIMENT_REGISTRY.keys()))
    exp_parser.add_argument('--model', '-m', help='which model to load', choices=tuple(MODEL_REGISTRY.keys()))

    exp_args, other_args = exp_parser.parse_known_args()

    exp_class = EXPERIMENT_REGISTRY[exp_args.experiment]
    #print('exp_class-----',exp_class)
    print('kkkkk----',exp_args.model)
    model_class = MODEL_REGISTRY[exp_args.model]
    print('model_class----',model_class)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    exp_class.add_arguments(experiment_group)

    model_group = parser.add_argument_group('model')
    model_class.add_arguments(model_group)

    args = parser.parse_args(other_args)

    if args.gpus is not None and isinstance(args.gpus, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        args.gpus = 1

    # TODO: push to lightning
    #args.gradient_clip_val = float(args.gradient_clip_val)

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        #print(group.title,group_dict,'\n')
        groups[group.title] = argparse.Namespace(**group_dict)

    lightning_args =  groups['pl.Trainer'] 
    print(lightning_args)              #groups['lightning_options']

    logger = TensorBoardLogger(lightning_args.default_root_dir, name=f'{exp_args.experiment}/{exp_args.model}')
    lightning_args.logger = logger


    hparams = groups['experiment']
    model_params = groups['model']

    for k, v in vars(model_params).items():
        setattr(hparams, k, v)
    
    print('lightning_args',lightning_args)
    
    trainer = Trainer.from_argparse_args(lightning_args)

    ## debugging 
    #trainer.fast_dev_run=False
    #trainer.callbacks=[TQDMProgressBar(refresh_rate=20)]



    #trainer.max_epochs=5
    print('updated trainer.fast_dev_run----------',trainer.fast_dev_run)

    #print('gpu',trainer.gpus,trainer.root_gpu,'root_device',trainer.strategy.root_device)
    #print(trainer.strategy)
    #print('fast_dev_run',trainer.fast_dev_run)
    print('model_params',vars(model_params))
    if not args.validate:
        warnings.filterwarnings('ignore', message='.*was not registered in the param store because.*', module=r'pyro\.primitives')
    model = model_class(**vars(model_params))
    print('model-------',model,'\n')
    experiment = exp_class(hparams, model)
    #print('experiment------',experiment,'\n')

trainer.fit(experiment)
