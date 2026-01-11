# import os, yaml
from rawnet.model import RawNet

# config_path = os.path.join(os.path.dirname(__file__), 'model_config_RawNet.yaml')
# with open(config_path, 'r') as f_yaml:
#     config = yaml.load(f_yaml)

# def CNN_Attention(device='cuda'):
#     return RawNet(config['model'], device)

def CNN_Attention(config: dict, device='cuda'):
    hypers: dict = {k: v for k, v in config.items()}
    hypers['filts'] = [list(item) if isinstance(item, tuple) else \
        item for item in hypers['filts']]
    hypers['blocks'] = list(hypers['blocks'])
    return RawNet(hypers, device)
