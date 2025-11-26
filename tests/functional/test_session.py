"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import pathlib

def test_session_attributes(session, config, land_column, irradiance, 
                             atmosphere_column):
    """
    Verify that the required attributes of Session exist and have the correct 
    types.

    Parameters
    ----------
    session : Session
        Instance of the Session class
    """
    
    assert hasattr(session, "config")
    assert isinstance(session.config, type(config))
    
    assert hasattr(session, "land_column")
    assert isinstance(session.land_column, type(land_column))
    
    assert hasattr(session, "solar_irradiance")
    assert isinstance(session.solar_irradiance, type(irradiance))
    
    assert hasattr(session, "atmosphere_column")
    assert isinstance(session.atmosphere_column, type(atmosphere_column))

    assert hasattr(session, "ROOT_PATH")
    assert isinstance(session.ROOT_PATH, pathlib.PosixPath)
    
    assert hasattr(session, "outputs")
    
    assert hasattr(session, "run")
    assert callable(getattr(session, "run"))

