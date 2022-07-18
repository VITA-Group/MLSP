import yaml
from easydict import EasyDict
import os

def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.out_path + '/' + args.exp_name, 'config.yaml')
        if not os.path.exists(cfg_path):
            print("Failed to resume, there is no {cfg_path}")
            raise FileNotFoundError()
        print(f'Resume yaml from {cfg_path}')
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume:
        if args.local_rank == 0:
            print(args.resume,args.local_rank)
            save_experiment_config(args, config, logger)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)        
    return config

def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def save_experiment_config(args, config, logger = None):
    config_path = args.out_path + '/' + args.exp_name
    config_path = os.path.join(config_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    print(f'Copy the Config file from {args.config} to {config_path}')