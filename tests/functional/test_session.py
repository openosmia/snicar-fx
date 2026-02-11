"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from pydantic import BaseModel


def test_session_attributes(session):
    """
    Verify that the required attributes of Session exist and have the correct
    types.

    Parameters
    ----------
    session : Session
        Instance of the Session class
    """

    assert hasattr(session, "config")
    assert isinstance(session.config, BaseModel)


def test_get_package_root(session):

    package_root = session.get_package_root()

    assert "snicar-fx" in package_root.parts


# def test_run(session):

#     outputs = session.run()

#     assert isinstance(outputs.BBA, float)
