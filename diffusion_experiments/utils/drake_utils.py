import numpy as np
from manipulation.station import Scenario
from pydrake.all import (
    LightParameter,
    Meshcat,
    MeshcatParams,
    Quaternion,
    RenderEngineVtkParams,
    RigidTransform,
    RollPitchYaw,
)


def change_camera_to_point_lighting(scenario: Scenario, main_camera_name: str = "camera0"):
    for camera_config in scenario.cameras.values():
        if camera_config.name == main_camera_name:
            camera_config.renderer_name = "RenderEngineVtk"
            this_params = RenderEngineVtkParams()
            this_params.lights = [LightParameter(type="point")]
            camera_config.renderer_class = this_params
        else:
            camera_config.renderer_name = "RenderEngineVtk"

    return scenario


def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)


def xyz_quaternion(xyz, quaternion):
    """Shorthand for defining a pose."""
    quaternion = np.asarray(quaternion)
    return RigidTransform(Quaternion(quaternion), xyz)


def ConfigureAndStartMeshcat(scenario: Scenario) -> Meshcat:
    params = MeshcatParams()

    param_tuples = [
        # Remove axes.
        dict(path="/Axes", property="visible", value=False),
        # Remove grid.
        dict(path="/Grid", property="visible", value=False),
    ]

    model_files = set(directive.add_model.file for directive in scenario.directives if directive.add_model is not None)
    if any("mujoco_floor" in file for file in model_files):
        param_tuples.extend(
            [
                # Set background colors.
                # https://github.com/google-deepmind/mujoco/blob/main/model/plugin/sdf/scene.xml
                dict(path="/Background/<object>", property="top_color", value=[0.3, 0.5, 0.7]),  # blue
                dict(path="/Background/<object>", property="bottom_color", value=[0.0, 0.0, 0.0]),  # black
            ]
        )
    elif any("drake_floor" in file for file in model_files):
        param_tuples.extend(
            [
                # Set background colors.
                dict(path="/Background/<object>", property="top_color", value=[0.7, 0.7, 0.7]),  # gray
                dict(path="/Background/<object>", property="bottom_color", value=[0.0, 0.0, 0.0]),  # black
            ]
        )
    elif any("gray_floor" in file for file in model_files):
        param_tuples.extend(
            [
                # Set background colors.
                dict(path="/Background/<object>", property="top_color", value=[0.7, 0.7, 0.7]),  # gray
                dict(path="/Background/<object>", property="bottom_color", value=[0.0, 0.0, 0.0]),  # black
            ]
        )

    # Set grid and axes properties
    params.initial_properties = [MeshcatParams.PropertyTuple(**t) for t in param_tuples]

    meshcat = Meshcat(params)

    # Set camera pose
    meshcat.SetCameraPose(
        camera_in_world=[1.5, 0, 1.0],
        target_in_world=[0.0, 0, 0.0],
    )

    # Add reflector below the floor.
    meshcat.SetObjectFromThreeJsCode(
        path="/drake/reflector",
        three_js_lambda="""
        () => {
            const geometry = new THREE.PlaneGeometry(20, 20);
            const reflector = new THREE_EXAMPLES.Reflector(geometry, {
                clipBias: 0.003,
                textureWidth: 1024,
                textureHeight: 1024,
                color: new THREE.Color(.2, 0.3, 0.4),
            });

            // Position slightly below the floor
            reflector.position.z = -1.01;

            return reflector;
        }
        """,
    )

    meshcat.EvalJavaScriptCode(
        """
        const toggle_pretty_mode = (is_pretty_on) => {
            const lightPaths = [
                ["Lights", "PointLightNegativeX", "<object>"],
                ["Lights", "PointLightPositiveX", "<object>"],
                ["Lights", "SpotLight", "<object>"],
                ["Lights", "FillLight", "<object>"],
                // ["Lights", "AmbientLight", "<object>"], <-- has no shadows
            ]

            if (is_pretty_on) {
                // Enable shadows.
                lightPaths.forEach(path => {
                    this.set_property(path, "castShadow", true);
                });

                // Enable reflector.
                this.set_property(["drake", "reflector"], "visible", true)

                // Disable contact forces.
                this.set_property(["drake", "contact_forces"], "visible", false)
            } else {
                lightPaths.forEach(path => {
                    this.set_property(path, "castShadow", false);
                });

                // Disable reflector.
                this.set_property(["drake", "reflector"], "visible", false)

                // Enable contact forces.
                this.set_property(["drake", "contact_forces"], "visible", true)
            }
        }
        toggle_pretty_mode(true);
        this.set_control("Pretty mode", toggle_pretty_mode, true);
        """
    )

    return meshcat
