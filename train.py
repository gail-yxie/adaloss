import argparse
import collections
import torch
import mlflow
import mlflow.pytorch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from collections import OrderedDict
from utils import set_seed
from utils import Adagradnorm


def log_params(conf: OrderedDict, parent_key: str = None):
    for key, value in conf.items():
        if parent_key is not None:
            combined_key = f'{parent_key}-{key}'
        else:
            combined_key = key

        if not isinstance(value, OrderedDict):
            mlflow.log_param(combined_key, value)
        else:
            log_params(value, combined_key)


def main(config: ConfigParser):
    """
    :param config: Configuration
    :type config: ConfigParser
    """
    logger = config.get_logger('train')

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=config['data_loader']['args']['shuffle'],
        validation_split=config['data_loader']['args']['validation_split'],
        training=True,
        num_workers=config['data_loader']['args']['num_workers'],
        pin_memory=config['data_loader']['args']['pin_memory']
    )

    valid_data_loader = data_loader.split_validation()

    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=128,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)
    model.cuda()

    # get function handles of loss and metrics
    logger.info(config.config)
    if hasattr(data_loader.dataset, 'num_raw_example'):
        num_examp = data_loader.dataset.num_raw_example
    else:
        num_examp = len(data_loader.dataset)

    train_loss = getattr(module_loss, config['train_loss']['type'])(num_examp=num_examp,
                                                                    num_classes=config['num_classes'],
                                                                    alpha=config['train_loss']['args']['alpha'])

    val_loss = getattr(module_loss, config['val_loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    if 'Adagradnorm' in config['optimizer']['type']:
        optimizer = Adagradnorm(trainable_params, lr=config['optimizer']['args']['lr'],
                                weight_decay=config['optimizer']['args']['weight_decay'],
                                momentum=config['optimizer']['args']['momentum'],
                                initial_accumulator_value=config['optimizer']['args']['initial_accumulator_value']**2)
    else:
        optimizer = config.initialize('optimizer', torch.optim, [{'params': trainable_params}])

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, train_loss, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      test_data_loader=test_data_loader,
                      lr_scheduler=lr_scheduler,
                      val_criterion=val_loss
                      )

    trainer.train()
    config.get_logger('trainer', config['trainer']['verbosity'])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--save-model',  default=0, type=int,
                      help='save model or not (default: not save 0)')                      
   
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['--mm', '--momentum'], type=float, target=('optimizer', 'args', 'momentum')),
        CustomArgs(['--wd', '--weight_decay'], type=float, target=('optimizer', 'args', 'weight_decay')),
        CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size')),
        CustomArgs(['--b0', '--b0'], type=float, target=('train_loss', 'args', 'b0')),
        CustomArgs(['--alpha', '--alpha'], type=float, target=('train_loss', 'args', 'alpha')),
        CustomArgs(['--cc', '--cc'], type=float, target=('train_loss', 'args', 'cc')),
        CustomArgs(['--name', '--exp_name'], type=str, target=('name',)),
        CustomArgs(['--seed', '--seed'], type=int, target=('seed',)),
        CustomArgs(['--ep', '--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--model', '--model'], type=str, target=('arch', 'type')),
        CustomArgs(['--adanormb0', '--normb0'], type=float, target=('optimizer', 'args', 'initial_accumulator_value'))
    ]
    config = ConfigParser.get_instance(args, options)

    # set random seed to get deterministic behavior
    set_seed(config['seed'])
    print(f"Set seed: {config['seed']}.")

    main(config)
