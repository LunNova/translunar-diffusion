import argparse, os, sys, datetime, glob, importlib, csv
import random
import time
import torch

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=True):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd['state_dict'] if 'state_dict' in pl_sd else pl_sd
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:', m)
    if len(u) > 0 and verbose:
        print('unexpected keys:', u)

    return model


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-n',
        '--name',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='postfix for logdir',
    )
    parser.add_argument(
        '-r',
        '--resume',
        type=str,
        const=True,
        default='',
        nargs='?',
        help='resume from logdir or checkpoint in logdir',
    )
    parser.add_argument(
        '-b',
        '--base',
        nargs='*',
        metavar='base_config.yaml',
        help='paths to base configs. Loaded from left-to-right. '
        'Parameters can be overwritten or added with command-line options of the form `--key value`.',
        default=list(),
    )
    parser.add_argument(
        '-t',
        '--train',
        type=str2bool,
        const=True,
        default=False,
        nargs='?',
        help='train',
    )
    parser.add_argument(
        '--no-test',
        type=str2bool,
        const=True,
        default=False,
        nargs='?',
        help='disable test',
    )
    parser.add_argument(
        '-p', '--project', help='name of new or path to existing project'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=None,
        help='seed for seed_everything',
    )
    parser.add_argument(
        '-f',
        '--postfix',
        type=str,
        default='',
        help='post-postfix for default name',
    )
    parser.add_argument(
        '-l',
        '--logdir',
        type=str,
        default='logs',
        help='directory for logging dat shit',
    )
    parser.add_argument(
        '--scale_lr',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='scale base-lr by ngpu * batch_size * n_accumulate',
    )

    parser.add_argument(
        '--datadir_in_name',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Prepend the final directory in the data_root to the output directory name',
    )

    parser.add_argument(
        '--actual_resume',
        type=str,
        default='',
        help='Path to model to resume from but without keeping steps/epoch/etc',
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to directory with training images',
    )

    parser.add_argument(
        '--embedding_manager_ckpt',
        type=str,
        default='',
        help='Initialize embedding manager from a checkpoint',
    )

    parser.add_argument(
        '--placeholder_tokens',
        type=str,
        nargs='+',
        default=['*'],
        help='Placeholder token which will be used to denote the concept in future prompts',
    )

    parser.add_argument(
        '--init_word',
        type=str,
        help='Word to use as source for initial token embedding.',
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def run():
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    resume_ckpt_path = None

    if opt.actual_resume and opt.resume:
        raise ValueError(
            '--actual_resume and -r/--resume cannot be specified both.'
        )

    if opt.name and opt.resume:
        raise ValueError(
            '-n/--name and -r/--resume cannot be specified both.'
            'If you want to resume training in a new log folder, '
            'use -n/--name in combination with --resume_from_checkpoint'
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError('Cannot find {}'.format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split('/')
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = '/'.join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip('/')

            # Resume from most recently changed checkpoint by default
            ckpt = max(
                glob.iglob(f'{logdir}/checkpoints/**.ckpt'),
                key=os.path.getctime,
            )
            assert os.path.exists(ckpt)

        _tmp = logdir.split('/')
        nowname = _tmp[-1]
        resume_ckpt_path = ckpt

        if opt.base:
            print(f'Using config from {opt.base}')
        else:
            print(
                f"Resuming with existing configs (specify --base if you don't want this)"
            )
            base_configs = sorted(
                glob.glob(os.path.join(logdir, 'configs/*.yaml'))
            )
            opt.base = base_configs + opt.base
    else:
        if opt.name:
            name = '_' + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = '_' + cfg_name
        else:
            name = ''

        if opt.datadir_in_name:
            now = os.path.basename(os.path.normpath(opt.data_root)) + now

        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, 'checkpoints')
    cfgdir = os.path.join(logdir, 'configs')
    if not opt.seed:
        opt.seed = random.randrange(0, 1 << 32)
    print(f'Using randomly selected global seed: {opt.seed}')
    seed_everything(opt.seed)

    trainer = None
    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop('lightning', OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get('trainer', OmegaConf.create())

        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not 'devices' in trainer_config:
            del trainer_config['accelerator']
            cpu = True
        else:
            gpuinfo = trainer_config['devices']
            print(f'Running on GPUs {gpuinfo}')
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        if 'personalization_config' in config.model.params:
            config.model.params.personalization_config.params.embedding_manager_ckpt = (
                opt.embedding_manager_ckpt
            )
            config.model.params.personalization_config.params.placeholder_tokens = (
                opt.placeholder_tokens
            )

            if opt.init_word:
                config.model.params.personalization_config.params.initializer_words[
                    0
                ] = opt.init_word

        if opt.actual_resume or resume_ckpt_path:
            model = load_model_from_config(
                config, opt.actual_resume or resume_ckpt_path
            )
        else:
            model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            'csv': {
                'target': 'pytorch_lightning.loggers.CSVLogger',
                'params': {
                    'name': 'csv',
                    'save_dir': logdir,
                },
            },
        }
        use_online_loggers = False
        if use_online_loggers:
            default_logger_cfgs.extend(
                {
                    'wandb': {
                        'target': 'pytorch_lightning.loggers.WandbLogger',
                        'params': {
                            'name': nowname,
                            'save_dir': logdir,
                            'offline': True,
                            'anonymous': True,
                            'id': nowname,
                            'log_model': False,
                        },
                    },
                    'comet': {
                        'target': 'pytorch_lightning.loggers.comet.CometLogger',
                        'params': {
                            'project_name': nowname,
                            'save_dir': logdir,
                        },
                    },
                }
            )
        if 'logger' in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        if 'target' in logger_cfg and isinstance(logger_cfg.target, str):
            logger_cfg = OmegaConf.create({'logger': logger_cfg})
        logger_cfg = OmegaConf.merge(default_logger_cfgs, logger_cfg)
        for k in logger_cfg.keys():
            logger = logger_cfg[k]
            if 'params' not in logger:
                logger.params = {}
            if 'save_dir' not in logger.params:
                logger.params.save_dir = logdir
            if 'name' not in logger.params:
                logger.params.name = nowname
        trainer_kwargs['logger'] = [
            instantiate_from_config(l) for k, l in logger_cfg.items()
        ]

        # add callback which sets up log directory
        default_callbacks_cfg = {
            'custom_checkpointing_callback': {
                'target': 'lun.callbacks.CustomCheckpointing',
                'params': {
                    'resume': opt.resume,
                    'now': now,
                    'logdir': logdir,
                    'ckptdir': ckptdir,
                    'cfgdir': cfgdir,
                    'config': config,
                    'lightning_config': lightning_config,
                },
            },
            'image_logger': {
                'target': 'lun.callbacks.ImageLogger',
                'params': {
                    'batch_frequency': 750,
                    'max_images': 4,
                    'clamp': True,
                },
            },
            'learning_rate_logger': {
                'target': 'pytorch_lightning.callbacks.LearningRateMonitor',
                'params': {
                    'logging_interval': 'step',
                    # "log_momentum": True
                },
            },
            'cuda_callback': {'target': 'lun.callbacks.EpochVram'},
        }

        callbacks_cfg = OmegaConf.merge(
            default_callbacks_cfg,
            lightning_config.get('callbacks') or OmegaConf.create(),
        )

        # set dirpath for all checkpoint callbacks
        for k, callback in callbacks_cfg.items():
            if (
                callback.target
                == 'pytorch_lightning.callbacks.ModelCheckpoint'
                and 'dirpath' not in callback.params
            ):
                callback.params.dirpath = ckptdir

        trainer_kwargs['callbacks'] = [
            instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
        ]
        for val in trainer_kwargs['callbacks']:
            print('Loaded callback ', type(val), val)
        if 'max_steps' in trainer_opt:
            trainer_kwargs['max_steps'] = trainer_opt.max_steps

        print(trainer_opt)
        print(trainer_kwargs)
        if 'strategy' in trainer_opt:
            strat = trainer_opt.strategy

            if strat == 'deepspeed':
                deepspeed_args = lightning_config.get(
                    'deepspeed', OmegaConf.create()
                )
                from pytorch_lightning.strategies import DeepSpeedStrategy

                strat = DeepSpeedStrategy(**deepspeed_args)
                print(f'Inited deepspeed strategy with args {deepspeed_args}')

            trainer_opt.strategy = strat

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        config.data.params.train.params.data_root = opt.data_root
        if 'validation' in config.data.params:
            config.data.params.validation.params.data_root = opt.data_root
        data = instantiate_from_config(config.data)
        print(data)

        # configure learning rate
        bs = config.data.params.batch_size
        base_lr = config.model.base_learning_rate
        ngpu = (
            1
            if cpu
            else len(lightning_config.trainer.devices.strip(',').split(','))
        )
        accumulate_grad_batches = lightning_config.trainer.get(
            'accumulate_grad_batches', 1
        )

        if accumulate_grad_batches > 1:
            print(
                f'accumulate_grad_batches = {accumulate_grad_batches}'
                "\nThis is known to cause poor training results for me and I'm not sure why!"
            )

        lr_scale_factor = accumulate_grad_batches * ngpu * bs

        print(
            'Calculated lr scale factor {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize)'.format(
                lr_scale_factor,
                accumulate_grad_batches,
                ngpu,
                bs,
            )
        )

        if opt.scale_lr:
            model.learning_rate = lr_scale_factor * base_lr
            print(
                'Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)'.format(
                    model.learning_rate,
                    accumulate_grad_batches,
                    ngpu,
                    bs,
                    base_lr,
                )
            )
        else:
            model.learning_rate = base_lr
            print('++++ NOT USING LR SCALING ++++')
            print(f'Setting learning rate to {model.learning_rate:.2e}')

        # run
        if opt.train:
            try:
                trainer.fit(model, data, ckpt_path=resume_ckpt_path)
            except Exception:
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    finally:
        if trainer and trainer.global_rank == 0:
            print(trainer.profiler.summary())
