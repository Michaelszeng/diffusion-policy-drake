import numpy as np
from pydrake.all import Diagram, DiagramBuilder

from planning_through_contact.simulation.planar_pushing.gamepad_controller import (
    GamepadController,
)


class GamepadControllerSource(Diagram):
    def __init__(
        self,
        meshcat,
        translation_scale: float,
        deadzone: float,
        gamepad_orientation: np.ndarray,
    ):
        super().__init__()

        builder = DiagramBuilder()

        # Gamepad Controller
        self._gamepad_controller = builder.AddNamedSystem(
            "GamepadController",
            GamepadController(
                meshcat=meshcat,
                translation_scale=translation_scale,
                deadzone=deadzone,
                gamepad_orientation=gamepad_orientation,
            ),
        )

        # Export inputs and outputs (external)
        builder.ExportInput(
            self._gamepad_controller.GetInputPort("pusher_pose_measured"),
            "pusher_pose_measured",
        )
        builder.ExportInput(self._gamepad_controller.GetInputPort("run_flag"), "run_flag")
        builder.ExportOutput(self._gamepad_controller.get_output_port(), "planar_position_command")

        builder.BuildInto(self)

    def get_button_values(self):
        return self._gamepad_controller.get_button_values()

    def reset(self, reset_xy=None):
        self._gamepad_controller.reset(reset_xy=None)

    def set_translation_scale(self, translation_scale):
        self._gamepad_controller.set_translation_scale(translation_scale)

    def get_translation_scale(self):
        return self._gamepad_controller.get_translation_scale()
