"""
This file is part of the snicar-fx software package.

https://github.com/openosmia/snicar-fx

"""

from ..components.land import LandColumn
from ..components.atmosphere import AtmosphereColumn
from ..components.solar import SolarIrradiance
from .config import Config
from ..solvers.two_stream_solver import solve_two_stream_rt
from ..solvers.multi_stream_solver import solve_multi_stream_rt


class Session:
    """
    Session driver orchestrating the input configuation, the
    different snicarfx objects, the solvers, run execution and output
    storage.

    It also provides a simple API to update certain input fields in
    the different components at run time.
    """

    def __init__(self, input_file: str):

        # parse configuration file
        self.config = Config.from_yaml(input_file)

        # build components
        self.land_column = LandColumn(self.config)
        self.solar_irradiance = SolarIrradiance(self.config)
        self.atmosphere_column = AtmosphereColumn(self.config)

        # store history of updates
        self._applied_updates = {
            "SOLVER": {},
            "SOLAR": {},
            "ATMOSPHERE": {},
            "LAND": {},
        }

        # save outputs
        self.outputs = None

    def _prepare_updates(self, kwargs, allowed_fields):
        forbidden = set(kwargs) - allowed_fields
        if forbidden:
            raise ValueError(
                f"Cannot update: {', '.join(forbidden)}. "
                "These fields cannot be changed at runtime. "
                "Please modify them in the input YAML file."
            )

        return {k: v for k, v in kwargs.items() if v is not None}

    def _write_current_state(self):
        """Write current session state to dictionnary, to be addded to
        the input file"""

        current_state = self.config.dict()

        # merge applied updates on top of the static config
        for section, section_updates in self._applied_updates.items():
            current_state[section].update(section_updates)

        return current_state

    def update_solver(self, *, validate=True, **kwargs):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {"TYPE"}
        updates = self._prepare_updates(kwargs, allowed_fields)

        # store applied updates
        self._applied_updates["SOLVER"].update(updates)

        # validate a copy of the config if requested (config is immutable)
        if validate:
            self.config.model_copy(update={"SOLVER": updates})

        # add what should be modified in SolarIrradiance due to SZA update
        if "TYPE" in updates:
            pass
            # self.solar_irradiance.SZA = updates["SZA"]
            # self.solar_irradiance.X()

    def update_solar(self, *, validate=True, **kwargs):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {"SZA"}
        updates = self._prepare_updates(kwargs, allowed_fields)

        # store applied updates
        self._applied_updates["SOLAR"].update(updates)

        # validate a copy of the config if requested (config is immutable)
        if validate:
            self.config.model_copy(update={"SOLAR": updates})

        # add what should be modified in SolarIrradiance due to SZA update
        if "SZA" in updates:
            self.solar_irradiance.SZA = updates["SZA"]
            self.solar_irradiance.X()

    def update_atmosphere(self, *, validate=True, **kwargs):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {"SKY_CONDITIONS"}
        updates = self._prepare_updates(kwargs, allowed_fields)

        # store applied updates
        self._applied_updates["ATMOSPHERE"].update(updates)

        # validate a copy of the config if requested (config is immutable)
        if validate:
            self.config.model_copy(update={"ATMOSPHERE": updates})

        # add what should be modified in SolarIrradiance due to SZA update
        if "SKY_CONDITIONS" in updates:
            pass
            # self.solar_irradiance.SZA = updates["SZA"]
            # self.solar_irradiance.X()

    def update_land(self, *, validate=True, **kwargs):
        """
        Update allowed solar fields.
        """

        # Keys that are allowed to be modified
        allowed_fields = {
            "THICKNESS",
            "LAYER_TYPE",
            "DENSITY",
            "SPECIFIC_SURFACE_AREA",
            "LWC",
            "GRAIN_SHAPE",
            "RF_TYPE",
            "SFC",
        }
        updates = self._prepare_updates(kwargs, allowed_fields)

        # store applied updates
        self._applied_updates["LAND"].update(updates)

        # validate a copy of the config if requested (config is immutable)
        if validate:
            self.config.model_copy(update={"LAND": updates})

        # add what should be modified in SolarIrradiance due to SZA update
        # if "SKY_CONDITIONS" in updates:
        # pass
        # self.solar_irradiance.SZA = updates["SZA"]
        # self.solar_irradiance.X()

    def run(self):
        """Run the radiative transfer solver.
        TODO: Add outputs to an existing DataFrame out outputs or create it"""

        if self.config.SOLVER.TYPE == "two-stream":
            self.outputs = solve_two_stream_rt(self.land_column, self.solar_irradiance)

        elif self.config.SOLVER.TYPE == "multi-stream":
            self.outputs = solve_multi_stream_rt(self.land_column, 
                                                 self.atmosphere_column,
                                                 self.solar_irradiance)

        return self.outputs
