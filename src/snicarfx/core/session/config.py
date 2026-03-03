"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import pathlib
from typing import Literal, Union, get_args

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    RootModel,
    confloat,
    conint,
    conlist,
    model_validator,
)
from pydantic.fields import PydanticUndefined


class Solver(BaseModel):
    """

    Define the valid ranges and types of the solver used in the configuration.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.

    """

    # radiative transfer solver to use
    TYPE: Literal["two-stream-ad", "multi-stream-ada", "multi-stream-disort"] = Field(
        description="Radiative transfer solver to use. two-stream uses the Delta-Eddington formulation, multi-stream uses the advanced matrix operator method and adding solver. See https://github.com/openosmia/snicar-fx?tab=readme-ov-file#references for details."
    )

    # explicit surface-atmosphere coupling
    ATMOSPHERE_COUPLING: bool = Field(
        description="If true, atmopshere layers are added on top of land layers and the extra-terrestrial solar irradiance at the top of atmosphere is used as boundary condition. If false, only land layers are modeled and the surface solar irradiance is used as boundary condition."
    )

    # levels to output
    OUTPUT_LEVELS: Literal["BOA", "TOA", "BOA+TOA"] = Field(
        description="Levels to output. If Top of Atmosphere (TOA) and ATMOSPHERE_COUPLING, only the upward loop of the solver is computed. If Bottom of Atmosphere (BOA)+TOA and ATMOSPHERE_COUPLING, both upward and downward loops are computed (slower)."
    )

    # number of streams to consider in solver
    N_STREAMS: int = Field(
        default=16, ge=12, le=100, description="Number of streams used by the solver."
    )

    N_LEGENDRE_MOMENTS: conint(ge=1, le=100) | None = Field(
        default=None,
        description="Number of Legendre moments to use in phase functions (<= N_STREAMS). Defaults to N_STREAMS (which defaults to 16).",
    )

    DELTA_SCALING: Literal["M", "M+"] | None = Field(
        default="M",
        description="Delta scaling to be applied. If M, delta-M scaling (Wiscombe 1977) is applied to single scattering properties by truncating the ice/snow phase function using the last legendre expansion coefficient. If M+, delta-M+ scaling (Lin et al. 2017) is applied to single scattering properties by truncating the ice/snow phase function.",
    )

    N_FOURIER_MODES: conint(ge=1, le=100) | None = Field(
        default=None,
        description="Number of Fourier modes to solve for azimuth dependency. Defaults to 1 (no azimuth dependency).",
    )

    AZIMUTH_ANGLES: (
        tuple[confloat(ge=0, le=360), confloat(ge=0, le=360), confloat(ge=0.01, le=360)]
        | None
    ) = Field(
        default=(0.0, 180.0, 20.0),
        description="Viewing azimuth angle (degrees).",
    )
        
    POLAR_ANGLES: (
        tuple[confloat(ge=0, le=90), confloat(ge=0, le=90), confloat(ge=0.01, le=50)]
        | None
    ) = Field(
        default=(5.0, 55.0, 5.0),
        description="Viewing polar angle (degrees).",
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_n_streams_even(self):
        if self.N_STREAMS % 2 != 0:
            raise ValueError(f"N_STREAMS must be even, got {self.N_STREAMS}")
        return self

    @model_validator(mode="after")
    def set_n_legendre_moments(self):
        # Default N_LEGENDRE_MOMENTS to N_STREAMS if not set
        if self.N_LEGENDRE_MOMENTS is None:
            self.N_LEGENDRE_MOMENTS = self.N_STREAMS

        # Validate it does not exceed N_STREAMS (see Chandrasekhar book)
        if self.N_LEGENDRE_MOMENTS > self.N_STREAMS:
            raise ValueError(
                f"N_LEGENDRE_MOMENTS cannot exceed " f"N_STREAMS ({self.N_STREAMS})"
            )

        return self

    @model_validator(mode="after")
    def set_n_fourier_modes(self):
        # Default N_FOURIER_MODES to 0 if not set
        if self.N_FOURIER_MODES is None:
            self.N_FOURIER_MODES = 1

        # Validate it does not exceed N_STREAMS (see Chandrasekhar book)
        if self.N_FOURIER_MODES > self.N_STREAMS:
            raise ValueError(f"N_FOURIER_MODES cannot exceed " f"N_STREAMS")

        return self

    @model_validator(mode="after")
    def check_azimuth_range(self):
        """
        Validate end > start and step < end - start.
        """

        start, end, step = self.AZIMUTH_ANGLES
        if end <= start:
            raise ValueError(
                f"AZIMUTH_ANGLES must be a valid range ([start, end, step]), with end ({end}) larger than start ({start})."
            )

        if step > (end - start):
            raise ValueError(
                f"AZIMUTH_ANGLES must be a valid range ([start, end, step]), with step ({step}) smaller than the difference between start and end ({end-start})."
            )

        return self

    @model_validator(mode="after")
    def check_type_atmosphere_coupling(self):
        if self.TYPE == "two-stream-ad" and self.ATMOSPHERE_COUPLING:
            raise ValueError(
                "SOLVER.ATMOSPHERE_COUPLING is not supported when SOLVER.TYPE='two-stream-ad'."
            )
        return self

    @model_validator(mode="after")
    def check_output_levels(self):
        if self.TYPE == "two-stream-ad" and "TOA" in self.OUTPUT_LEVELS:
            raise ValueError(
                "TOA output level is not supported when SOLVER.TYPE='two-stream-ad'."
            )

        if not self.ATMOSPHERE_COUPLING and "TOA" in self.OUTPUT_LEVELS:
            raise ValueError(
                "TOA output level is not supported when SOLVER.ATMOSPHERE_COUPLING=False."
            )
        return self


class Spectral(BaseModel):
    """

    Define the valid ranges and types of the spectral settings.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.
    """

    # Mode for spectral calculations
    MODE: Literal[
        "monochromatic",
        "band-srf-integration",
        "band-snicar-default",
        "band-solar-weighted-mean",
    ] = Field(
        description="Mode to use for spectral calculations. `monochromatic` solves for discrete wavelengths with virtually infinitesimal band widths. `band-srf-integration` solves at a high 1cm-1 resolution within each band before integrating using the Spectral Response Functions (SRF) of the specified satellite platform. `band-snicar-default` applies an unweighted band average to high-resolution (1cm-1) optical properties of the atmosphere and solar components, and selects the center wavelength of the high-resolution (1cm-1) optical properties of the land component, before solve. `band-solar-weighted-mean` applies an solar-weighted band average to high-resolution (1cm-1) optical properties of all components before solve."
    )

    # spectral range (start, end, step) or satellite instrument
    RESOLUTION: (
        tuple[
            confloat(ge=200, le=5000),
            confloat(ge=200, le=5000),
            confloat(ge=0.001, le=100),
        ]
        | Literal["SENTINEL-3-OLCI", "PRISMA-HYC", "ENVISAT-MERIS"]
    ) = Field(
        description="The spectral resolution to cover. If a satellite platform is passed, then all bands are solved for."
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_spectral_range(self):
        """
        If SPECTRAL_RESOLUTION is numeric (start, end, step), validate
        end > start and step < end - start.
        Skip validation if it's a satellite platform string.
        """

        if isinstance(self.RESOLUTION, tuple):
            start, end, step = self.RESOLUTION
            if end <= start:
                raise ValueError(
                    f"SPECTRAL_RESOLUTION must be a valid spectral range ([start, end, step]), with end ({end}) larger than start ({start})."
                )

            if step > (end - start):
                raise ValueError(
                    f"SPECTRAL_RESOLUTION must be a valid spectral range ([start, end, step]), with step ({step}) smaller than the difference between start and end ({end-start})."
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
    
    # Solar Zenith Angle (unit: degrees)
    SAA: int = Field(..., ge=0, le=360, description="The Solar Azimuth Angle (SAA).")

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class IntegratedGasConcentrations(BaseModel):
    """
    Define an object for the configuration of all gas concentrations.

    Inherits from the Pydantic BaseModel class, which enables automatic type
    validation and parsing of yaml files.

    """

    # O3 concentration used to scale the atmospheric profile
    O3: confloat(ge=0.0) | None = Field(
        default=None,
        description=(
            "Ozone concentration used to scale the atmospheric profile."
            "If a numeric value is provided (in kg.m-2), it is used to scale the O3 profile. "
        ),
    )

    # # O2 concentration used to scale the atmospheric profile
    O2: confloat(ge=0.0) | None = Field(
        default=None,
        description=(
            "Oxygen concentration used to scale the atmospheric profile."
            "If a numeric value is provided (in kg.m-2), it is used to scale the O2 profile. "
        ),
    )

    # H2O concentration used to scale the atmospheric profile
    H2O: confloat(ge=0.0) | None = Field(
        default=None,
        description=(
            "Water wapor concentration used to scale the atmospheric profile."
            "If a numeric value is provided (in kg.m-2), it is used to scale the H2O profile. "
        ),
    )

    # CO2 concentration used to scale the atmospheric profile
    CO2: confloat(ge=0.0) | None = Field(
        default=None,
        description=(
            "Carbon dioxide concentration used to scale the atmospheric profile."
            "If a numeric value is provided (in kg.m-2), it is used to scale the CO2 profile. "
        ),
    )

    # NO2 concentration used to scale the atmospheric profile
    NO2: confloat(ge=0.0) | None = Field(
        default=None,
        description=(
            "Nitrogen dioxide concentration used to scale the atmospheric profile."
            "If a numeric value is provided (in kg.m-2), it is used to scale the NO2 profile. "
        ),
    )

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

    # Aerosol properties to be used
    AEROSOL_PROPERTIES: str | None = Field(
        default=None,
        pattern=r".*\.(nc|csv)$",
        description="File containing the optical properties of aerosols.",
    )

    # AOD used to scale aerosol optical depth
    INTEGRATED_AOD_550: confloat(ge=0.0, le=10.0) | None = Field(
        default=0.0,
        description=(
            "Aerosol Optical Depth (AOD) at 550nm integrated over the atmosphere column."
        ),
    )

    # all provided gas concentrations
    INTEGRATED_GAS_CONCENTRATIONS: IntegratedGasConcentrations | None = None

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

    @model_validator(mode="after")
    def check_grain_layer_compatibility(self):
        """
        GRAIN_SHAPE can't be Robledano et al. 2023 (1) if
        LAYER_TYPE is solid ice (1 or 2).

        """

        grain_shapes = np.array(self.GRAIN_SHAPE)
        layer_types = np.array(self.LAYER_TYPE)

        invalid = (layer_types != 0) & (grain_shapes == 1)

        if np.any(invalid):

            invalid_layers = np.where(invalid)[0] + 1

            raise ValueError(
                f"GRAIN_SHAPE must be sphere (0) if solid ice LAYER_TYPE is used (1 or 2). Layers {invalid_layers} do not meet this criterion."
            )

        return self


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
    _ROOT_PATH: pathlib.Path | None = PrivateAttr(default=None)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_solver_atmosphere_compatibility(self):
        if (
            self.ATMOSPHERE.SKY_CONDITIONS == "cloudy"
        ):
            raise ValueError(
                "ATMOSPHERE.SKY_CONDITIONS='cloudy' is not supported for now. "
            )
        return self

    @model_validator(mode="after")
    def check_solver_layer_type_compatibility(self):
        if ("multi-stream" in self.SOLVER.TYPE and 1 in self.LAND.LAYER_TYPE):
            raise ValueError(
                "Fresnel boundaries are not supported when " "SOLVER.TYPE='multi-stream'."
            )
        return self

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
        """
        Print a prettier version of the model scheme including:
          - regular fields
          - nested BaseModels
        """

        model = model or Config
        prefix = "  " * indent

        # Regular fields
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
                isinstance(a, (str, int)) for a in typ.__args__
            ):
                enum_values = list(typ.__args__)

            # Default value
            default = field.default
            if default is None or default is PydanticUndefined:
                default_str = "required"
            else:
                default_str = default

            # Build line
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
