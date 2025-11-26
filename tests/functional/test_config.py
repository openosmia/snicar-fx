"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from pydantic import BaseModel

def test_config_attributes(config):
    """
    Verify that the required attributes of Config exist and have the correct 
    types.

    Parameters
    ----------
    config : Config
        Instance of the Config class
    """
    
    assert isinstance(config, BaseModel)
    
    assert hasattr(config, "SOLVER")
    
    assert hasattr(config, "SOLAR")
    
    assert hasattr(config, "ATMOSPHERE")
    
    assert hasattr(config, "LAND")

    assert hasattr(config, "from_yaml")
    assert callable(getattr(config, "from_yaml"))
    
    assert hasattr(config, "check_lengths")
    assert callable(getattr(config, "check_lengths"))
    
    assert config.model_config["extra"] == "forbid"

