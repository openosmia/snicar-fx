#!/usr/bin/env python3
"""

@author: snicar-fx team

"""


from typing import Literal

import yaml
from pydantic import BaseModel, Field, confloat, conint, conlist, model_validator


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
    RTM: Rtm
    ICE: Ice
    LIGHT_ABSORBING_PARTICLES: LightAbsorbingParticles

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_lengths(self):
        """
        Check that all ICE and LIGHT_ABSORBING_PARTICLES
        layer-related lists have the same length

        """

        ice_lists = [
            self.ICE.THICKNESS,
            self.ICE.LAYER_TYPE,
            self.ICE.DENSITY,
            self.ICE.SPECIFIC_SURFACE_AREA,
            self.ICE.LWC,
            self.ICE.GRAIN_SHAPE,
        ]
        lengths = {len(lst) for lst in ice_lists}

        if len(lengths) > 1:
            raise ValueError(
                f"All ICE layer-related lists must have the same length, got lengths: "
                f"{[len(lst) for lst in ice_lists]}"
            )

        ice_layers = len(self.ICE.THICKNESS)

        # Check that all particle CONC lists match ICE layers
        for particle_name in ["BC", "ALG"]:
            particle = getattr(self.LIGHT_ABSORBING_PARTICLES, particle_name)
            if particle is not None and len(particle.CONC) != ice_layers:
                raise ValueError(
                    f"Particle {particle_name} CONC list length ({len(particle.CONC)}) "
                    f"does not match number of ice layers ({ice_layers})"
                )

        return self


# read input data
with open("/home/adrien/research/snicar-fx/src/snicarfx/inputs.yaml") as ymlfile:
    input_data = yaml.load(ymlfile, Loader=yaml.FullLoader)

# validate input file
config = Config.model_validate(input_data)
