"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

import pathlib
from typing import Literal, Union, get_args, get_origin
import numpy as np
import yaml
from pydantic import (
    BaseModel,
    RootModel,
    Field,
    PrivateAttr,
    confloat,
    conint,
    conlist,
    model_validator,
)
import textwrap
from pydantic.fields import PydanticUndefined
from types import UnionType
import warnings


class Solver(BaseModel):
    """
    Pydantic BaseModel class defining the valid ranges and types of the solver
    configuration.
    """

    # radiative transfer solver to use
    TYPE: Literal["two-stream-ad", "multi-stream-ada", "multi-stream-disort"] = Field(
        description="Radiative transfer solver to use",
        examples="'two-stream-ad' selects the two-stream Delta-Eddington formulation, 'multi-stream-ada' selects the multi-stream solver with advanced matrix operator and adding method, and 'multi-stream-disort' selects the DISORT solver via the PythonicDISORT package. See https://github.com/openosmia/snicar-fx?tab=readme-ov-file#references for references.",
    )

    # explicit surface-atmosphere coupling
    ATMOSPHERE_COUPLING: bool = Field(
        description="Explicit surface-atmosphere coupling",
        examples="If True, atmosphere layers are added on top of land layers and the extra-terrestrial solar irradiance at the top of atmosphere is used as boundary condition. if False, only land layers are modeled and the surface solar irradiance is used as boundary condition. Can only be True using multi-stream solvers.",
    )

    # levels to output
    OUTPUT_LEVELS: Literal["BOA", "TOA", "BOA+TOA", "TOA+BOA"] = Field(
        description="Levels at which to return radiance/reflectance",
        examples="'BOA' = Bottom of Atmosphere, 'TOA' = Top of Atmosphere, 'BOA+TOA' or 'TOA+BOA' = both. 'TOA' can only be included if ATMOSPHERE_COUPLING is True.",
    )

    DELTA_SCALING: Literal["M", "M+"] = Field(
        default="M",
        description="Type of Delta scaling to apply",
        examples="'M' = delta-M scaling (Wiscombe 1977), 'M+' = delta-M+ scaling (Lin et al. 2017). Must be set to 'M' if TYPE = 'two-stream-ad'. Automatically uses the IMS-TMS correction if 'M' and TYPE = 'multi-stream-disort'.",
    )

    N_STREAMS: int = Field(
        default=None,
        ge=8,
        le=100,
        description="Number of streams used by multi-stream solver",
        examples="Only used with multi-stream solvers. If no value set in the input file (= None) and multi-stream solver used, set to 16.",
    )

    N_LEGENDRE_MOMENTS: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of Legendre moments to use in phase functions",
        examples="Only used with multi-stream solvers. Must be <= N_STREAMS. If no value set in the input file (= None), the value is set to N_STREAMS.",
    )

    N_FOURIER_MODES: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of Fourier modes (for azimuth dependency)",
        examples="Only used with multi-stream solvers. Must be <= N_STREAMS. Default is no azimuth dependency (= 1), which works with all solvers.",
    )

    POLAR_ANGLES: (
        tuple[confloat(ge=1, le=89), confloat(ge=1, le=89), confloat(ge=1, le=89)]
        | None
    ) = Field(
        default=None,
        description="Viewing polar angle (degrees)",
        examples="Only used with multi-stream-disort solver, as the multi-stream-ada solver uses a fixed array of viewing polar angles. Must be prescribed as a range (start, end, step). If no value set in the input file (= None), set to (5.0, 55.0, 5.0).",
    )

    AZIMUTH_ANGLES: (
        tuple[confloat(ge=0, le=360), confloat(ge=0, le=360), confloat(ge=1, le=359)]
        | None
    ) = Field(
        default=None,
        description="Viewing azimuth angle (degrees)",
        examples="Only used with multi-stream solvers when N_FOURIER_MODES > 1. Must be prescribed as a range (start, end, step). If no value set in the input file (= None), set to (20.0, 180.0, 20.0).",
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_atmosphere_coupling(self):
        """
        Verify that atmosphere coupling is False when using the two-stream
        solver.
        """

        if self.TYPE == "two-stream-ad" and self.ATMOSPHERE_COUPLING:
            raise ValueError(
                "Atmosphere coupling is not supported with the two-stream solver (TYPE = 'two-stream-ad')."
            )
        return self

    @model_validator(mode="after")
    def check_output_levels(self):
        """
        Verify that top-of-atmosphere outputs are not requested for uncoupled
        simulations / with the two-stream solver.
        """

        if "TOA" in self.OUTPUT_LEVELS:
            if self.TYPE == "two-stream-ad":
                raise ValueError(
                    "TOA output level is not supported with the two-stream solver (TYPE = 'two-stream-ad')."
                )

            if not self.ATMOSPHERE_COUPLING:
                raise ValueError(
                    "TOA output level is not supported without atmosphere coupling (ATMOSPHERE_COUPLING = False)."
                )
        return self

    @model_validator(mode="after")
    def check_delta_scaling(self):
        """
        Verify that Delta-M+ scaling is not requested when using the two-stream
        solver.
        """

        if self.TYPE == "two-stream-ad" and "+" in self.DELTA_SCALING:
            raise ValueError(
                "Delta-M+ scaling is not available with two-stream solver (TYPE = 'two-stream-ad')."
            )

        return self

    @model_validator(mode="after")
    def set_streams_and_legendre_moments(self):
        """
        Set number of streams and Legendre moments to default if not read from
        the input file, else check that the user-defined values are valid, i.e.
        even number of streams and number Legendre moments lower or equal to
        number of streams (see Chandrasekhar, Radiative Transfer, 1950).
        """

        if "multi-stream" in self.TYPE:
            if self.N_STREAMS is None:
                self.N_STREAMS = 16
            if self.N_LEGENDRE_MOMENTS is None:
                self.N_LEGENDRE_MOMENTS = self.N_STREAMS
        else:
            self.N_STREAMS = 2

        if self.N_STREAMS is not None and self.N_STREAMS % 2 != 0:
            raise ValueError(f"N_STREAMS must be even, but received {self.N_STREAMS}")

        if self.N_LEGENDRE_MOMENTS is not None and self.TYPE == "two-stream-ad":
            raise ValueError(
                "Legendre decomposition not available with two-stream solver (TYPE = 'two-stream-ad') so N_LEGENDRE_MOMENTS cannot be used."
            )

        if "multi-stream" in self.TYPE and self.N_LEGENDRE_MOMENTS > self.N_STREAMS:
            raise ValueError(
                f"N_LEGENDRE_MOMENTS cannot exceed N_STREAMS ({self.N_STREAMS})"
            )

        return self

    @model_validator(mode="after")
    def check_n_fourier_modes(self):
        """
        Verify that the number of Fourier modes does not exceed the number of
        streams (see Chandrasekhar, Radiative Transfer, 1950) and does not
        exceed 1 with the two-stream solver (no azimuth dependency).
        """

        if self.N_FOURIER_MODES > 1 and self.TYPE == "two-stream-ad":
            raise ValueError(
                "Fourier modes are not available with two-stream solver (TYPE = 'two-stream-ad')."
            )

        if self.N_FOURIER_MODES > self.N_STREAMS:
            raise ValueError(
                f"N_FOURIER_MODES cannot exceed N_STREAMS (currently {self.N_STREAMS})"
            )

        return self

    @model_validator(mode="after")
    def set_polar_angles(self):
        """
        Set polar angle array to default if not provided in the input file,
        and check that no polar angle array exists when using the
        two-stream solver (no polar angle dependency).
        """

        if self.POLAR_ANGLES is None and self.TYPE == "multi-stream-disort":
            self.POLAR_ANGLES = (5.0, 55.0, 5.0)

        if self.POLAR_ANGLES is not None and self.TYPE == "two-stream-ad":
            raise ValueError(
                "Polar angle resolution not available with two-stream solver (TYPE = 'two-stream-ad')."
            )

        return self

    @model_validator(mode="after")
    def check_polar_angle_range(self):
        """
        Verify that the range of polar angles provided is valid, ie:
        end > start and step < end - start.
        """
        if self.POLAR_ANGLES is not None:
            start, end, step = self.POLAR_ANGLES
            if end <= start:
                raise ValueError(
                    "POLAR_ANGLES must be a valid range ([start, end, step]), with end larger than start, but received end ({end}) <= start ({start})."
                )

            if step > (end - start):
                raise ValueError(
                    f"POLAR_ANGLES must be a valid range ([start, end, step]), with step smaller than the total range, but received step ({step}) > range ({end-start})."
                )

        return self

    @model_validator(mode="after")
    def set_azimuth_angles(self):
        """
        Set azimuth angle array to default if not provided in the input file,
        and check that no azimuth angle array exists when the number of Fourier
        modes is 1 or when using the two-stream solver (no polar angle dependency).
        """

        if (
            self.AZIMUTH_ANGLES is None
            and "multi-stream" in self.TYPE
            and self.N_FOURIER_MODES > 1
        ):
            self.AZIMUTH_ANGLES = (20.0, 180.0, 20)

        if self.AZIMUTH_ANGLES is not None and self.TYPE == "two-stream-ad":
            raise ValueError(
                "Azimuth angle resolution not available with two-stream solver (TYPE = 'two-stream-ad')."
            )
        if self.AZIMUTH_ANGLES is not None and self.N_FOURIER_MODES == 1:
            raise ValueError(
                "Azimuth angle resolution not available with only one Fourier mode (N_FOURIER_MODES = 1)."
            )
        return self

    @model_validator(mode="after")
    def check_azimuth_angle_range(self):
        """
        Verify that the range of azimuth angles provided is valid, ie:
        end > start and step < end - start.
        """

        if self.AZIMUTH_ANGLES is not None:
            start, end, step = self.AZIMUTH_ANGLES
            if end <= start:
                raise ValueError(
                    "AZIMUTH_ANGLES must be a valid range ([start, end, step]), with end larger than start, but received end ({end}) <= start ({start})."
                )

            if step > (end - start):
                raise ValueError(
                    f"AZIMUTH_ANGLES must be a valid range ([start, end, step]), with step smaller than the total range, but received step ({step}) > range ({end-start})."
                )

        return self


class Spectral(BaseModel):
    """
    Pydantic BaseModel class defining the valid ranges and types of the spectral
    configuration.
    """

    MODE: Literal[
        "monochromatic",
        "band-snicar-default",
        "band-srf-solar-weighted-mean",
        "band-srf-weighted-mean",
        "band-srf-integration",
    ] = Field(
        description="Type of spectral mode in calculations",
        examples="'monochromatic' solves and returns the output at discrete wavelengths, while all other modes return bands. 'band-snicar-default' is the default mode of the original SNICAR model (band means for the solar irradiance, center wavelength for land properties). 'band-srf-solar-weighted-mean' applies a weighted integration to atmosphere/land/solar properties using the solar irradiance and satellite response function for each band, except for the Legendre moments which are taken at the central wavelength. 'band-srf-weighted-mean' applies a weighted integration to atmosphere/land/solar properties using the satellite response function for each band, except for the Legendre moments which are taken at the central wavelength. 'band-srf-integration' applies the satellite response function after solving at 1cm-1 resolution (! it is computationally very expensive).",
    )

    RESOLUTION: (
        tuple[
            confloat(ge=200, le=5000),
            confloat(ge=200, le=5000),
            Union[confloat(ge=0.001, le=100), Literal["1cm-1"]],
        ]
        | Literal["SENTINEL-3-OLCI", "PRISMA-HYC", "ENVISAT-MERIS"]
    ) = Field(
        description="Spectral resolution of the output (continuous or sensor-based range)",
        examples="If a tuple is passed, the output is returned for each band or each monochromatic wavelength in the range. (!) the spectral range is restricted to 300 - 2700nm when using coupled simulations. For satellite platforms, the output is returned either for each band or for each wavelength within the satellite sensor reponse function.",
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_spectral_range_tuple(self):
        """
        Verify that the spectral resolution provided as a tuple is valid,
        ie:
            - end > start and step < end - start.
            - the range is not too large when using coupled simulations
        """

        if isinstance(self.RESOLUTION, tuple):
            start, end, step = self.RESOLUTION
            if end <= start:
                raise ValueError(
                    "SPECTRAL_RESOLUTION must be a valid range ([start, end, step]), with end larger than start, but received end ({end}) <= start ({start})."
                )

            if isinstance(step, float) and step > (end - start):
                raise ValueError(
                    f"SPECTRAL_RESOLUTION must be a valid range ([start, end, step]), with step smaller than the total range, but received step ({step}) > range ({end-start})."
                )

        return self

    @model_validator(mode="after")
    def check_spectral_mode(self):
        """
        Verify band mode compatibility:
            - spectral resolution is a satellite platform if the spectral
            mode is integrating over a spectral response function.
            - spectral mode is monochromatic if resolution is tuple with cm-1
        """

        if "srf" in self.MODE and isinstance(self.RESOLUTION, tuple):
            raise ValueError(
                "The spectral resolution must be a satellite platform, not a tuple/range when using band spectral modes with SRF integration."
            )

        if (
            isinstance(self.RESOLUTION, tuple)
            and isinstance(self.RESOLUTION[2], str)
            and self.MODE != "monochromatic"
        ):
            raise ValueError(
                "The spectral mode must be monochromatic when the resolution is in cm-1."
            )
        return self


class Solar(BaseModel):
    """
    Pydantic BaseModel class defining the valid ranges and types of the solar
    configuration.
    """

    SZA: int = Field(..., ge=0, le=89, description="Solar zenith angle")

    SAA: int | None = Field(
        default=None,
        ge=0,
        le=360,
        description="Solar azimuth angle.",
        examples="Only used for multi-stream solvers if azimuth dependency is calculated (i.e. number of Fourier modes > 1).",
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class IntegratedGasConcentrations(BaseModel):
    """
    Pydantic BaseModel class defining the valid ranges and types of the
    integrated gas concentrations.
    """

    O3: float | None = Field(
        default=None,
        ge=0.0,
        le=0.01,
        description=("Column-integrated O3 concentration"),
        examples="Only used for coupled simulations. The value is in standard units of the CAMS product (in kg.m-2) and is used to scale the O3 profile.",
    )

    H2O: float | None = Field(
        default=None,
        ge=0.0,
        le=50,
        description=("Column-integrated H2O concentration"),
        examples="Only used for coupled simulations. The value is in standard units of the CAMS product (in kg.m-2) and is used to scale the H2O profile.",
    )

    NO2: float | None = Field(
        default=None,
        ge=0.0,
        le=2e-6,
        description=("Column-integrated NO2 concentration"),
        examples="Only used for coupled simulations. The value is in standard units of the CAMS product (in kg.m-2) and is used to scale the NO2 profile.",
    )

    CO2: float | None = Field(
        default=None,
        ge=0.0,
        le=10,
        description=("Column-integrated CO2 concentration"),
        examples="Only used for coupled simulations. The value is in standard units of the CAMS product (in kg.m-2) and is used to scale the CO2 profile.",
    )

    O2: float | None = Field(
        default=None,
        ge=0.0,
        le=5000,
        description=("Column-integrated O2 concentration"),
        examples="Only used for coupled simulations. The value is in standard units of the CAMS product (in kg.m-2) and is used to scale the O2 profile.",
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class Atmosphere(BaseModel):
    """
    Pydantic BaseModel class defining the valid ranges and types of the
    atmosphere configuration.
    """

    SKY_CONDITIONS: Literal["clear", "clear_fully_direct", "cloudy"] = Field(
        default="clear",
        description="Sky conditions determining type of surface irradiance",
        examples="Only clear sky conditions are available for now. This field is mostly used for uncoupled simulations to determine the ratio of direct/diffuse radiation arriving at the surface (for coupled simulation, the TOA irradiance is always fully direct). 'clear_fully_direct' assumes 100% direct irradiance, 'clear' represents direct solar beam dominance, 'cloudy' is fully diffuse irradiance.",
    )

    ATMOSPHERIC_PROFILE_TYPE: Literal["afglss", "afglss_downscaled", "test"] = Field(
        description="Name of standard atmospheric profile",
        examples="This field selects a type of atmospheric profile (AFGL Atmospheric Constituent Profiles developed by Anderson et al. 1986), which includes pressure, temperature, air density, as well as O3, H2O, CO2 and NO2 concentrations for each atmospheric level. For now only the Subarctic Summer (afglss) profile is available. The downscaled version corresponds to a similar profile with ~2x less layers. Test is a two-layer atmosphere for testing purposes only.",
    )

    AEROSOL_PROPERTIES: str | None = Field(
        default=None,
        pattern=r".*\.(nc|csv)$",
        description="File name for the optical properties of a given aerosol mixture",
        examples="Only used for coupled simulations. The file must include the single scattering properties of aerosols (e.g. standard OPAC files).",
    )

    INTEGRATED_AOD_550: float | None = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description=("Aerosol optical depth (AOD) at 550nm"),
        examples="Only used for couple simulations. The AOD value is integrated over the atmosphere column such as the standard CAMS product.",
    )

    INTEGRATED_GAS_CONCENTRATIONS: IntegratedGasConcentrations | None = None

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_sky_conditions(self):
        """
        Verify that cloudy sky conditions is not selected as it is not supported
        for now.
        """

        if self.SKY_CONDITIONS == "cloudy":
            raise ValueError("Cloudy sky conditions are not available for now.")
        return self

    @model_validator(mode="after")
    def check_aerosol_file(self):
        """
        Verify that an aerosol file is given if AOD > 0.
        """

        if self.INTEGRATED_AOD_550 > 0 and self.AEROSOL_PROPERTIES is None:
            raise ValueError(
                "An aerosol file is required to set an aerosol optical thickness."
            )
        return self


class Particle(BaseModel):
    """
    Pydantic BaseModel class defining the valid ranges and types for a
    particle.
    """

    FILE: str = Field(
        pattern=r".*\.(nc|csv)$",
        description="File name for the optical properties of a given particle type",
        examples="Example: 'ice_algae.nc'",
    )

    CONC: conlist(confloat(ge=0.0, le=1e7), min_length=1, max_length=100) = Field(
        description="Concentration of a given particle type (ng/g)",
        examples="The concentration is expressed per g of ice for each layer and thus the absolute amount depends on the density.",
    )

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}


class LightAbsorbingParticles(RootModel[dict[str, Particle]]):
    """
    Pydantic RootModel class encapsulating the different Particle instances.
    """


class Land(BaseModel):
    """
    Pydantic BaseModel class defining the valid ranges and types for a
    the land configuration.
    """

    THICKNESS: conlist(confloat(ge=1e-5, le=1000), min_length=1, max_length=100) = (
        Field(description="Thickness of each vertical layer (meters)")
    )

    LAYER_TYPE: conlist(conint(ge=0, le=2), min_length=1, max_length=100) = Field(
        description="Type of each vertical layer",
        examples="0 is ice spheres, 1 is solid ice with Fresnel layer above and 2 is solid ice without Fresnel layer. (!) Fresnel layers are not available with multi-stream solvers.",
    )

    DENSITY: conlist(confloat(ge=10, le=924), min_length=1, max_length=100) = Field(
        description="Density of each vertical layer (kg m-3)",
        examples="(!) The density corresponds to the bulk medium of solid ice and liquid water, so the upper bound of the density range varies depending on the liquid water content. Without liquid water, the maximum density is 916.999 kg m-3, and this maximum increases when liquid water is added.",
    )

    SPECIFIC_SURFACE_AREA: conlist(
        confloat(ge=1e-5, le=25), min_length=1, max_length=100
    ) = Field(
        description="Specific surface area of each vertical layer (m2 kg-1)",
        examples="For ice surfaces (layer type > 0), the specific surface area will be directly related to the radius of a spherical air bubble. For snow surfaces, the specific surface area will be directly related to the radius of a spherical ice grain if the grain shape if spherical.",
    )

    LWC: conlist(confloat(ge=0.0, le=0.9), min_length=1, max_length=100) = Field(
        description="Liquid water content (LWC) in each vertical layer",
        examples="The liquid water content is modelled by mixing the absorption coefficients of ice and water for both snow and ice surfaces.",
    )

    RF_TYPE: Literal["Pic16", "Wrn08", "Coop21"] = Field(
        description="Source of ice refractive index",
        examples="The ice refractive index used is either from Picard et al. 2016, Warren and Brandt 2008, or Cooper et al. 2021. See https://github.com/openosmia/snicar-fx?tab=readme-ov-file#references for details.",
    )

    SFC: float = Field(
        ge=0.0,
        le=1.0,
        description="Reflectance of lower boundary of ice/snow column",
        examples="The reflectance is assumed Lambertian and constant with the wavelength.",
    )

    GRAIN_SHAPE: conlist(conint(ge=0, le=1), min_length=1, max_length=100) = Field(
        description="Ice grain or air bubble shape",
        examples="0 is sphere, 1 corresponds to the 'optical shape' from Robledano et al 2023. For ice layers (layer type > 0), only the spherical shape is available. See https://github.com/openosmia/snicar-fx?tab=readme-ov-file#references for details.",
    )

    ALTITUDE: float = Field(ge=0.0, le=10.0, description="Surface altitude (km)")

    LIGHT_ABSORBING_PARTICLES: LightAbsorbingParticles | None = None

    # only fields validated here are allowed
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_grain_layer_compatibility(self):
        """
        Verify that the grain shape is 0 for ice layers (layer type > 0).
        """

        grain_shapes = np.array(self.GRAIN_SHAPE)
        layer_types = np.array(self.LAYER_TYPE)

        invalid = (layer_types != 0) & (grain_shapes == 1)

        if np.any(invalid):

            invalid_layers = np.where(invalid)[0] + 1

            raise ValueError(
                f"Only spherical grain shape (=0) are allowed for ice layers (layer type > 1). Layers {invalid_layers} do not meet this criterion."
            )

        return self

    @model_validator(mode="after")
    def check_ice_and_air_fraction_validity(self):
        """
        Verify physical validity of ice and air fractions (>1 and >0, respectively).
        """

        ice_volume_fraction = (np.array(self.DENSITY) - np.array(self.LWC) * 1000) / 917
        air_volume_fraction = 1 - ice_volume_fraction - np.array(self.LWC)

        if any(ice_volume_fraction >= 1):
            invalid_layers = np.where(ice_volume_fraction >= 1)[0]
            raise ValueError(
                f"Volume ice fraction invalid. Please reduce density in layers {invalid_layers}."
            )

        if any(air_volume_fraction <= 0):
            invalid_layers = np.where(air_volume_fraction <= 0)[0]
            raise ValueError(
                f"Volume air fraction invalid. Please reduce density in layers {invalid_layers}."
            )

        return self


class Config(BaseModel):
    """
    Pydantic BaseModel class combining the SOLVER, SPECTRAL, SOLAR, ATMOSPHERE
    and LAND models and defining the valid combinations across them.
    """

    SOLVER: Solver
    SPECTRAL: Spectral
    SOLAR: Solar
    ATMOSPHERE: Atmosphere
    LAND: Land

    _wavelengths_land: np.ndarray | None = PrivateAttr(default=None)
    _wavelengths_solar: np.ndarray | None = PrivateAttr(default=None)
    _wavelengths_atmosphere: np.ndarray | None = PrivateAttr(default=None)
    _ROOT_PATH: pathlib.Path | None = PrivateAttr(default=None)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def check_saa_requirement(self):
        """
        Verify that azimuth angle array is only provided if azimuth dependency
        requested (i.e. multi stream solver + number of Fourier modes > 1).
        """

        needs_azimuth = ("multi-stream" in self.SOLVER.TYPE) and (
            self.SOLVER.N_FOURIER_MODES > 1
        )

        if needs_azimuth and self.SOLAR.SAA is None:
            raise ValueError(
                f"Solar azimuth angle is required if N_FOURIER_MODES > 1 (current: {self.SOLVER.N_FOURIER_MODES}). "
                "Please provide SAA in the SOLAR section."
            )
        if self.SOLAR.SAA is not None and needs_azimuth is False:
            raise ValueError(
                "Solar azimuth angle can only be prescribed with multi-stream solvers if N_FOURIER_MODES > 1."
                "Please remove the SAA from the input file or adapt solver/number of Fourier modes."
            )
        return self

    @model_validator(mode="after")
    def check_solver_layer_type_compatibility(self):
        """
        Verify that no Fresnel layer requested with a multi-stream solver.
        """

        if "multi-stream" in self.SOLVER.TYPE and 1 in self.LAND.LAYER_TYPE:
            raise ValueError(
                "Fresnel boundaries are not supported with multi-stream solvers."
            )
        return self

    @model_validator(mode="after")
    def check_solver_band_mode_compatibility(self):
        """
        Check compatibility between spectral mode and atmosphere coupling.
        """

        if (
            self.SPECTRAL.MODE == "band-snicar-default"
            and self.SOLVER.ATMOSPHERE_COUPLING
        ):
            raise ValueError(
                f"{self.SPECTRAL.MODE} spectral mode cannot be selected with atmosphere coupling."
            )
        return self

    @model_validator(mode="after")
    def check_atmosphere_fields_only_for_coupled(self):
        """
        Verify that atmosphere fields used for coupled simulations are None
        for uncoupled simulations.
        """

        if not self.SOLVER.ATMOSPHERE_COUPLING and (
            self.ATMOSPHERE.AEROSOL_PROPERTIES is not None
            or self.ATMOSPHERE.INTEGRATED_GAS_CONCENTRATIONS is not None
        ):
            raise ValueError(
                "Aerosol and gas properties cannot be prescribed in uncoupled simulations."
            )
        return self

    @model_validator(mode="after")
    def check_lengths(self):
        """
        Check that all layer-related properties of the LAND BaseModel
        have the same length (i.e. same number of layers everywhere).
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

    @model_validator(mode="after")
    def check_spectral_range_validity(self):
        """
        Verify that the spectral bounds provided as a tuple are valid
        depending on the model configuration, ie:
            - range is within 300-2700nm in all cases except for a non-coupled
              simulation with two-stream-ad solver, as the
              Legendre moments get >1 with HG function above 2700nm
        """

        if isinstance(self.SPECTRAL.RESOLUTION, tuple):
            start, end, step = self.SPECTRAL.RESOLUTION

            if self.SOLVER.TYPE != "two-stream-ad":
                if start < 300 or end > 2700:
                    raise ValueError(
                        "SPECTRAL_RESOLUTION must cover a valid range. Please modify SPECTRAL_RESOLUTION to be within 300-2700nm."
                    )

        return self

    @model_validator(mode="after")
    def check_irradiance_type_ada(self):
        """
        Verify that fully clear conditions are selected when using ADA.
        """

        if (
            self.ATMOSPHERE.SKY_CONDITIONS != "clear_fully_direct"
            and self.SOLVER.TYPE == "multi-stream-ada"
            and self.SOLVER.ATMOSPHERE_COUPLING == False
        ):
            raise ValueError(
                "The surface irradiance is currently treated as 100% direct "
                "when the ADA multi-stream solver is used. Please use "
                "clear_fully_direct as SKY_CONDITIONS."
            )

        return self

    # def raise_warnings(self):
    #     """
    #     Print warnings regarding physical assumptions made based on the
    #     selected configuration (e.g., irradiance treatment).
    #     """

    #     if (
    #         self.ATMOSPHERE.SKY_CONDITIONS == "clear"
    #         and self.SOLVER != "two-stream-ad"
    #         and self.SOLVER.ATMOSPHERE_COUPLING == False
    #     ):
    #         warnings.warn(
    #             "The irradiance is currently treated as 100% direct beam when "
    #             "the multi-stream solvers are used.",
    #             UserWarning,
    #             stacklevel=2,
    #         )

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Config":
        """Load YAML and trigger physical assumption warnings."""
        with open(yaml_file) as f:
            input_data = yaml.load(f, Loader=yaml.FullLoader)

        config_instance = cls.model_validate(input_data)

        # config_instance.raise_warnings()

        return config_instance

    @staticmethod
    def print_help(model: type[BaseModel] = None, indent: int = 0):
        """
        Print information about the fields that can be prescribed in the input
        file (description, type, allowed values, default values and usage).
        """
        model = model or Config
        c_indent = "  " * indent
        s_indent = "  " * (indent + 2)
        w = 80 - len(s_indent)

        # get the type of variable (string, int, list etc)
        def _fmt(t):
            o = get_origin(t)
            if o is Literal:
                return "string"
            if o in (Union, UnionType):
                p = []
                for a in get_args(t):
                    if a is type(None):
                        continue
                    if get_origin(a) is Literal:
                        p.append("string")
                    else:
                        p.append(a.__name__ if hasattr(a, "__name__") else str(a))
                return " or ".join(p) or "None"
            return o.__name__ if o else (t.__name__ if isinstance(t, type) else str(t))

        # get the bounds if given
        def _get_bounds(t):
            mn, mx = None, None
            for m in getattr(t, "__metadata__", []):
                if hasattr(m, "ge") and m.ge is not None:
                    mn = m.ge if mn is None else min(mn, m.ge)
                if hasattr(m, "gt") and m.gt is not None:
                    mn = m.gt if mn is None else min(mn, m.gt)
                if hasattr(m, "le") and m.le is not None:
                    mx = m.le if mx is None else max(mx, m.le)
                if hasattr(m, "lt") and m.lt is not None:
                    mx = m.lt if mx is None else max(mx, m.lt)
            return mn, mx

        # get the allowed range and values if given
        def _allowed(f, t):
            def _b(x):
                return _get_bounds(x)

            o = get_origin(t)
            types = [
                x
                for x in (get_args(t) if o in (Union, UnionType) else [t])
                if x is not type(None)
            ]
            if bool in types:
                return "[True, False]"

            cons, lits = [], []
            for x in types:
                xo = get_origin(x)
                if xo is list:
                    args = get_args(x)
                    if args:
                        mn, mx = _b(args[0])
                        rng = (
                            f"{mn} - {mx}"
                            if mn is not None and mx is not None
                            else "any"
                        )
                        lmin, lmax = None, None
                        for m in getattr(x, "__metadata__", []):
                            if hasattr(m, "min_length"):
                                lmin = m.min_length
                            if hasattr(m, "max_length"):
                                lmax = m.max_length
                        if lmin is None or lmax is None:
                            for m in f.metadata:
                                if hasattr(m, "min_length"):
                                    lmin = m.min_length
                                if hasattr(m, "max_length"):
                                    lmax = m.max_length
                        cons.append(
                            f"{rng} (Length: {lmin}-{lmax})"
                            if lmin is not None and lmax is not None
                            else f"{rng} (list)"
                        )
                elif xo is tuple:
                    parts = []
                    valid = False
                    for it in get_args(x):
                        mn, mx = _b(it)
                        if mn is not None and mx is not None:
                            parts.append(f"{mn}-{mx}")
                            valid = True
                        else:
                            parts.append("?")
                    if valid:
                        cons.append(f"({', '.join(parts)})")
                elif xo is Literal:
                    lits.extend(get_args(x))

            if not cons and not lits:
                mn, mx = None, None
                for m in f.metadata:
                    if hasattr(m, "ge") and m.ge is not None:
                        mn = m.ge if mn is None else min(mn, m.ge)
                    if hasattr(m, "gt") and m.gt is not None:
                        mn = m.gt if mn is None else min(mn, m.gt)
                    if hasattr(m, "le") and m.le is not None:
                        mx = m.le if mx is None else max(mx, m.le)
                    if hasattr(m, "lt") and m.lt is not None:
                        mx = m.lt if mx is None else max(mx, m.lt)
                if mn is not None and mx is not None:
                    return f"{mn} - {mx}"
                if mn is not None:
                    return f"value >= {mn}"
                if mx is not None:
                    return f"value <= {mx}"
            if lits:
                cons.append(str(lits))
            return " or ".join(cons) if cons else None

        # loop over all model fields (then prints all info if sub-field,
        # and print only name if upper-level field)
        for name, field in model.model_fields.items():
            o = get_origin(field.annotation)
            is_nested, target, is_root = False, None, False
            for c in (
                get_args(field.annotation)
                if o in (Union, UnionType)
                else [field.annotation]
            ):
                if isinstance(c, type):
                    if issubclass(c, RootModel):
                        is_nested, is_root = True, True
                        target = globals().get("Particle")
                        break
                    if issubclass(c, BaseModel):
                        is_nested, target = True, c
                        break

            print(f"\n{c_indent}{name}\n{c_indent}{'-'*len(name)}")

            # directly descriptions and values if no pydantic model inside
            # (ie subfield)
            if not is_nested:
                if field.description:
                    print(
                        textwrap.fill(
                            field.description,
                            width=w,
                            initial_indent=f"{s_indent}Description: ",
                            subsequent_indent=f"{s_indent}             ",
                        )
                    )
                print(f"{s_indent}Type: {_fmt(field.annotation)}")
                al = _allowed(field, field.annotation)
                if al:
                    print(
                        textwrap.fill(
                            f"Allowed values: {al}",
                            width=w,
                            initial_indent=s_indent,
                            subsequent_indent=f"{s_indent}       ",
                        )
                    )
                print(
                    f"{s_indent}Default: {'None' if field.default in (None, PydanticUndefined) else field.default}"
                )
                if field.examples:
                    ex = (
                        "; ".join(map(str, field.examples))
                        if isinstance(field.examples, list)
                        else str(field.examples)
                    )
                    print(
                        textwrap.fill(
                            f"Usage: {ex}",
                            width=w,
                            initial_indent=s_indent,
                            subsequent_indent=f"{s_indent}       ",
                        )
                    )

            # do not print description if pydantic model inside (ie upper level)
            # except for light absorbing particle field that has Particle models inside
            elif target:
                # special case for LAP fiedl
                if is_root and target.__name__ == "Particle":
                    print(f"\n{s_indent}PARTICLE_NAME\n{s_indent}-------------")
                    for pn, pf in target.model_fields.items():
                        ps = "  " * (indent + 2)
                        pw = 80 - len(ps)
                        print(f"\n{ps}{pn}\n{ps}{'-'*len(pn)}")
                        if pf.description:
                            print(
                                textwrap.fill(
                                    pf.description,
                                    width=pw,
                                    initial_indent=f"{ps}Description: ",
                                    subsequent_indent=f"{ps}             ",
                                )
                            )
                        print(f"{ps}Type: {_fmt(pf.annotation)}")
                        al = _allowed(pf, pf.annotation)
                        if al:
                            print(
                                textwrap.fill(
                                    f"Allowed values: {al}",
                                    width=pw,
                                    initial_indent=ps,
                                    subsequent_indent=f"{ps}       ",
                                )
                            )
                        print(
                            f"{ps}Default: {'None' if pf.default in (None, PydanticUndefined) else pf.default}"
                        )
                        if pf.examples:
                            ex = (
                                "; ".join(map(str, pf.examples))
                                if isinstance(pf.examples, list)
                                else str(pf.examples)
                            )
                            print(
                                textwrap.fill(
                                    f"Usage: {ex}",
                                    width=pw,
                                    initial_indent=ps,
                                    subsequent_indent=f"{ps}       ",
                                )
                            )
                # normal upper level field
                else:
                    Config.print_help(target, indent + 1)
