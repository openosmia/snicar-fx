"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from typing import Literal, Tuple, Union, get_args
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
from pydantic.fields import PydanticUndefined


class Solver(BaseModel):
    """

    Define the valid ranges and types of the solver used in the configuration.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.

    """

    # radiative transfer solver to use
    TYPE: Literal["two-stream", "multi-stream"] = Field(
        description="Radiative transfer solver to use. two-stream uses the Delta-Eddington formulation, multi-stream uses the advanced matrix operator method and adding solver. See https://github.com/openosmia/snicar-fx?tab=readme-ov-file#references for details."
    )

    # explicit surface-atmosphere coupling
    ATMOSPHERE_COUPLING: bool = Field(
        description="If true, atmopshere layers are added on top of land layers and the extra-terrestrial solar irradiance at the top of atmosphere is used as boundary condition. If false, only land layers are modeled and the surface solar irradiance is used as boundary condition."
    )

    # number of Legendre moments to use in phase functions
    N_LEGENDRE_MOMENTS: int = Field(
        default=15,
        ge=10,
        le=17,
        description="Number of Legendre moments to use in phase functions",
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Spectral(BaseModel):
    """

    Define the valid ranges and types of the spectral settings.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.
    """

    # Mode for spectral calculations
    MODE: Literal["monochromatic", "band"] = Field(
        description="Mode to use for spectral calculations. monochromatic solves for discrete wavelengths with virtually no band widths. band solves for bands that are combined using the specified BAND_METHOD."
    )

    # spectral range (start, end, step) or satellite instrument
    RESOLUTION: Union[
        Tuple[
            confloat(ge=200, le=5000),
            confloat(ge=200, le=5000),
            confloat(ge=0.001, le=100),
        ],
        Literal["SENTINEL-3-OLCI"],
    ] = Field(
        description="The spectral resolution to cover. If a satellite platform is passed, then all bands are solved for."
    )

    # Computation method for band mode
    BAND_METHOD: (
        Literal["srf-integration", "snicar-default", "solar-weighted-mean"] | None
    ) = Field(
        description="The computation method to apply in case MODE is band. srf-integration solves at a high 1cm-1 resolution within each band before being integrated using the Spectral Response Functions (SRF) of the specified satellite platform. center-wavelength only solves on the nominal center wavelength of the satellite platform. chandrasekhar-mean computes mean optical properties within sub-bands of each bands before solve."
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_second_greater(self):
        """
        If SPECTRAL_RANGE is numeric (start, end, step), validate end > start.
        Skip validation if it's a satellite platform string.
        """
        if isinstance(self.RESOLUTION, tuple):
            start, end, step = self.RESOLUTION
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
    SZA: int = Field(..., ge=0, le=89, description="The Solar Zenigh Angle (SZA).")

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Atmosphere(BaseModel):
    """

    Define the valid ranges and types of the atmosphere properties.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.
    """

    # sky conditions to apply direct- or diffuse-dominated radiation
    SKY_CONDITIONS: Literal["clear", "cloudy"] = Field(
        description="Sky conditions to appy direct- or diffuse-dominated radiation."
    )

    # type of atmospheric profile
    ATMOSPHERIC_PROFILE_TYPE: Literal["afglss"] = Field(
        description="Type of atmospheric profile to use. It includes elevation, pressure, temperature, air density, as well as O3, H2O, CO2 and NO2 concentrations."
    )

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
    FILE: str = Field(
        ...,
        pattern=r".*\.(nc|csv)$",
        description="File containing the optical properties of a given light absorbing particle.",
    )

    # light absorbing particle concentration
    CONC: conlist(confloat(ge=0.0, le=1e10), min_length=1, max_length=1000) = Field(
        description="Concentration of a given light absorbing particle in ..."
    )

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
    THICKNESS: conlist(confloat(ge=1e-10, le=1000), min_length=1, max_length=1000) = (
        Field(description="Thickness of each vertical layer in meters.")
    )

    # 0: ice spheres, 1: solid ice w/frsnl, 2: w/out frsnl
    LAYER_TYPE: conlist(Literal[0, 1, 2], min_length=1, max_length=1000) = Field(
        description="if 0, ice spheres. if 1, solide ice with Fresnel layer. If 2, solid ice without Fresnel layer."
    )

    # density of each layer (unit : kg m-3)
    DENSITY: conlist(
        confloat(ge=10, le=925),
        min_length=1,
        max_length=1000,
    ) = Field(description="Density of each vertical layer in kgm-3.")

    # m2 kg-1
    SPECIFIC_SURFACE_AREA: conlist(
        confloat(ge=1e-10, le=100), min_length=1, max_length=1000
    ) = Field(description="Specific surface area of each vertical layer in m2kg-1.")

    # LWC content in snow/ice
    LWC: conlist(confloat(ge=0.0, le=1.0), min_length=1, max_length=1000) = Field(
        description="Liquid Water content (LWC) in snow/ice"
    )

    # source of refraction index
    RF_TYPE: Literal["Pic16", "Wrn08", "Coop21"] = Field(
        description="Source of refraction index. See https://github.com/openosmia/snicar-fx?tab=readme-ov-file#references for details."
    )

    # reflectance of lower boundary
    SFC: float = Field(
        ..., ge=0.0, le=1.0, description="Reflectance of the lower boundary."
    )

    # grain shape: 0 is sphere, 1 is Robledano et al. 2023
    GRAIN_SHAPE: conlist(Literal[0, 1], min_length=1, max_length=1000) = Field(
        description="Grain shape. 0 is sphere, 1 is Robledano et al 2023. See https://github.com/openosmia/snicar-fx?tab=readme-ov-file#references for details."
    )

    # surface altitude (km)
    ALTITUDE: float = Field(..., ge=0.0, le=9.0, description="Surface altitude in km.")

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
    SPECTRAL: Spectral
    SOLAR: Solar
    ATMOSPHERE: Atmosphere
    LAND: Land

    # Private runtime-only attribute
    _wavelengths_land: np.ndarray | None = PrivateAttr(default=None)
    _wavelengths_solar: np.ndarray | None = PrivateAttr(default=None)
    _wavelengths_atmosphere: np.ndarray | None = PrivateAttr(default=None)
    # _wavelengths_srf: np.ndarray | None = PrivateAttr(default=None)
    # _spectral_response_function: np.ndarray | None = PrivateAttr(default=None)
    # _band_ranges: np.ndarray | None = PrivateAttr(default=None)
    _ROOT_PATH: pathlib.Path | None = PrivateAttr(default=None)

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

    @staticmethod
    def print_help(model: type[BaseModel] = None, indent: int = 0):
        """Print a prettier version of the model scheme"""

        model = model or Config
        prefix = "  " * indent
        for name, field in model.model_fields.items():
            typ = field.annotation

            # Simplify type display
            if hasattr(typ, "__origin__"):
                origin = typ.__origin__
                args = get_args(typ)
                if origin is Union:
                    typ_str = " | ".join(
                        t.__name__ if hasattr(t, "__name__") else str(t) for t in args
                    )
                else:
                    typ_str = origin.__name__
            elif isinstance(typ, type):
                typ_str = typ.__name__
            else:
                typ_str = str(typ)

            # Handle Literals / Enums
            enum_values = []
            if hasattr(typ, "__args__") and all(
                isinstance(a, str) or isinstance(a, int) for a in typ.__args__
            ):
                enum_values = list(typ.__args__)

            # Default value (replace PydanticUndefined with "required")
            default = field.default
            if default is None or default is PydanticUndefined:
                default_str = "required"
            else:
                default_str = default

            # Build help line
            line = f"{prefix}{name} ({typ_str}) [default: {default_str}]"
            if enum_values:
                line += f" Options: {enum_values}"
            if field.description:
                line += f" Description: '{field.description}'"
            print(line)
            print("\n")

            # Recurse into nested BaseModels
            if isinstance(typ, type) and issubclass(typ, BaseModel):
                Config.print_help(typ, indent + 1)
