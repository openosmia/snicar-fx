"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from typing import Literal, Tuple, Union
import numpy as np

import yaml
import pathlib
from pydantic import (
    BaseModel,
    Field,
    RootModel,
    confloat,
    conlist,
    model_validator,
    PrivateAttr,
)


class Solver(BaseModel):
    """

    Define the valid ranges and types of the solver used in the configuration.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.

    """

    # radiative transfer solver to use
    TYPE: Literal["two-stream", "multi-stream"]

    # spectral range (start, end, step) or satellite instrument
    SPECTRAL_RANGE: Union[
        Tuple[
            confloat(ge=200, le=5000),
            confloat(ge=200, le=5000),
            confloat(ge=1, le=100),
        ],
        Literal["SENTINEL-3-OLCI"],
    ]

    # explicit surface-atmosphere coupling
    ATMOSPHERE_COUPLING: bool

    # number of Legendre moments to use in phase functions
    N_LEGENDRE_MOMENTS: int = Field(15, ge=10, le=17)

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_second_greater(self):
        """
        If SPECTRAL_RANGE is numeric (start, end, step), validate end > start.
        Skip validation if it's a satellite platform string.
        """
        if isinstance(self.SPECTRAL_RANGE, tuple):
            start, end, step = self.SPECTRAL_RANGE
            if end <= start:
                raise ValueError(
                    f"SPECTRAL_RANGE second value ({end}) must be larger "
                    f"than the first ({start})"
                )

        return self


class Solar(BaseModel):
    """

    Define the valid ranges and types of the solar properties.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.
    """

    # Solar Zenith Angle (unit: degrees)
    SZA: int = Field(..., ge=0, le=89)

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Atmosphere(BaseModel):
    """

    Define the valid ranges and types of the atmosphere properties.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.
    """

    # sky conditions to apply direct- or diffuse-dominated radiation
    SKY_CONDITIONS: Literal["clear", "cloudy"]

    # type of atmospheric profile
    ATMOSPHERIC_PROFILE_TYPE: Literal["afglss"]

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Particle(BaseModel):
    """
    Define the valid ranges and types for the properties of a light absorbing
    particle.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.

    """

    # check that file name has netCDF or csv extension
    FILE: str = Field(..., pattern=r".*\.(nc|csv)$")

    # light absorbing particle concentration
    CONC: conlist(confloat(ge=0.0, le=1e10), min_length=1, max_length=1000)

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class LightAbsorbingParticles(RootModel[dict[str, Particle]]):
    """
    Define an object for the configuration of all light absorbing particles.
    Ensure each key in LIGHT_ABSORBING_PARTICLES is valid.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.

    """


class Land(BaseModel):
    """
    Define the valid ranges and types of the land surface properties.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.

    """

    # thickness of each vertical layer (unit : m)
    THICKNESS: conlist(confloat(ge=1e-10, le=1000), min_length=1, max_length=1000)

    # 0: ice spheres, 1: solid ice w/frsnl, 2: w/out frsnl
    LAYER_TYPE: conlist(Literal[0, 1, 2], min_length=1, max_length=1000)

    # density of each layer (unit : kg m-3)
    DENSITY: conlist(confloat(ge=10, le=925), min_length=1, max_length=1000)

    # m2 kg-1
    SPECIFIC_SURFACE_AREA: conlist(
        confloat(ge=1e-10, le=100), min_length=1, max_length=1000
    )

    # LWC content in snow/ice
    LWC: conlist(confloat(ge=0.0, le=1.0), min_length=1, max_length=1000)

    # source of refraction index
    RF_TYPE: Literal["Pic16", "Wrn08", "Coop21"]

    # reflectance of lower boundary
    SFC: float = Field(..., ge=0.0, le=1.0)

    # grain shape: 0 is sphere, 1 is Robledano et al. 2023
    GRAIN_SHAPE: conlist(Literal[0, 1], min_length=1, max_length=1000)

    # surface altitude (km)
    ALTITUDE: float = Field(..., ge=0.0, le=9.0)

    # all provided light absorbing particles
    LIGHT_ABSORBING_PARTICLES: LightAbsorbingParticles | None = None

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Config(BaseModel):
    """
    Combine the different objects inheriting from BaseModel.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.

    """

    SOLVER: Solver
    SOLAR: Solar
    ATMOSPHERE: Atmosphere
    LAND: Land

    # Private runtime-only attribute
    _wavelengths: np.ndarray | None = PrivateAttr(default=None)
    _ROOT_PATH: pathlib.Path | None = PrivateAttr(default=None)
    _spectral_response_function: np.ndarray | None = PrivateAttr(default=None)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_lengths(self):
        """
        Check that all LAND and LIGHT_ABSORBING_PARTICLES
        layer-related lists have the same length

        """

        land_lists = [
            self.LAND.THICKNESS,
            self.LAND.LAYER_TYPE,
            self.LAND.DENSITY,
            self.LAND.SPECIFIC_SURFACE_AREA,
            self.LAND.LWC,
            self.LAND.GRAIN_SHAPE,
        ]
        lengths = {len(lst) for lst in land_lists}

        if len(lengths) > 1:
            raise ValueError(
                f"All LAND layer-related lists must have the same length, but now"
                f"are: {[len(lst) for lst in land_lists]}"
            )

        land_layers = len(self.LAND.THICKNESS)

        # Check that all particle CONC lists match LAND layers
        laps = getattr(self.LAND, "LIGHT_ABSORBING_PARTICLES", None)
        if laps is not None:
            for particle_name, particle in laps.root.items():  # <-- use .root
                if particle is not None and len(particle.CONC) != land_layers:
                    raise ValueError(
                        f"Particle '{particle_name}' CONC list length "
                        f"({len(particle.CONC)}) does not match number of land "
                        f"layers ({land_layers})"
                    )

        return self

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Config":
        """Wrap the yaml file validation"""
        with open(yaml_file) as f:
            input_data = yaml.load(f, Loader=yaml.FullLoader)
        return cls.model_validate(input_data)
