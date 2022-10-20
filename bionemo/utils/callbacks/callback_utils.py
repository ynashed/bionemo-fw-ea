from pytorch_lightning import Callback

from typing import List, Optional

import bionemo.utils.callbacks
from bionemo.utils import lookup_or_use

__all__ = ["setup_callbacks"]


def _instantiate_callback(callback_cfg, parent_cfg, plugins: Optional[List] = None) -> Callback:
    """
    Instantiates callback
    
    Params
        callback_cfg: Config for callback to be created with 'class' attribute
        parent_cfg: Global cfg dict
        plugins: Optional plugins to be passed to callback
        
    Returns
        Instantiated class given in callback_cfg
    """
    callback_class = callback_cfg['class']
    cb = lookup_or_use(bionemo.utils.callbacks, callback_class, callback_cfg, parent_cfg, plugins)
    return cb

def _select_validation_callbacks(cfg) -> List:
    """
    Selects validation callbacks from config
    
    Params
        cfg: Dict
    
    Returns:
        List of selected validation callback config dicts
    """
    valid_cbs = []
    if 'validation' in cfg.model and cfg.model.validation['validation_enabled']:
        valid_cbs = [dset for dset in cfg.model.validation.datasets if dset['enabled']]
    return valid_cbs

def setup_callbacks(cfg, plugins: Optional[List] = None) -> List:
    """
    Sets up callbacks from cfg
    
    Params
        cfg: Dict
        plugins: Optional plugins to be passed to callbacks
        
    Returns
        List of Callbacks to be passed into Trainer
    """
    callbacks = []
    callbacks.extend(_select_validation_callbacks(cfg))
    
    callbacks = [_instantiate_callback(cb, cfg, plugins) for cb in callbacks]
    return callbacks
