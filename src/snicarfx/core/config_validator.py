#!/usr/bin/env python3
"""

@author: snicar-fx team

"""


from typing import Literal

import yaml
from pydantic import BaseModel, Field, confloat, conint, conlist


class Rtm(BaseModel):

    # radiation (0 is direct 1 is diffuse)
    DIRECT: Literal[0, 1]

    # first wavelength (unit: nm)
    WVL_START: int = Field(..., ge=200, le=5000)

    # last wavelength (unit: nm)
    WVL_END: int = Field(..., ge=200, le=5000)

    # spectral resolution (unit: nm)
    RESOLUTION: int = Field(..., ge=1, le=100)

    # Solar Zenith Angle (unit: degrees)
    SZA: int = Field(..., ge=40, le=70)

    # type of irradiance
    IRRADIANCE_TYPE: Literal["mls", "mlw", "saw", "sas", "smm", "hmn", "trp"]

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Ice(BaseModel):

    # thickness of each vertical layer (unit : m)
    THICKNESS: conlist(conint(ge=1, le=100), min_length=1, max_length=5)

    # 0: ice spheres, 1: solid ice w/frsnl, 2: w/out frsnl
    LAYER_TYPE: conlist(Literal[0, 1, 2], min_length=1, max_length=5)

    # density of each layer (unit : kg m-3)
    DENSITY: conlist(confloat(ge=200, le=1000), min_length=1, max_length=5)

    # source of refraction index
    RF_TYPE: Literal["Pic16", "Wrn08", "Coop21"]

    # m2 kg-1
    SPECIFIC_SURFACE_AREA: conlist(confloat(ge=0.1, le=100), min_length=1, max_length=5)

    # LWC content in snow/ice
    LWC: conlist(confloat(ge=0.0, le=1.0), min_length=1, max_length=5)

    # reflectance of lower boundary
    SFC: float = Field(..., ge=0.0, le=1.0)

    # grain shape: 0 is sphere, 1 is Robledano et al. 2023
    GRAIN_SHAPE: conlist(Literal[0, 1], min_length=1, max_length=5)

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Particle(BaseModel):
    FILE: str
    COATED: bool
    UNIT: int
    CONC: conlist(item_type=Literal[0, 1, 2], min_length=1, max_length=5)

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class LightAbsorbingParticles(BaseModel):
    # Optional: Black carbon and algae
    BC: Particle | None = None
    ALG: Particle | None = None

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Config(BaseModel):

    # Top level structure of the input file
    RTM: Rtm
    ICE: Ice
    LIGHT_ABSORBING_PARTICLES: LightAbsorbingParticles

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


# read input data
with open("/home/adrien/research/snicar-fx/src/snicarfx/inputs.yaml") as ymlfile:
    input_data = yaml.load(ymlfile, Loader=yaml.FullLoader)

config = Config.model_validate(input_data)
