"""Direct Omniverse simulation script showing SO-101 robot on a brown ground plane, with red cube and grey box (swapped), setup_cam on the right side looking left, and gripper camera."""

import sys
sys.path.append('/home/zozo/leisaac/source/leisaac')

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View the SO-101 standalone setup environment.")
parser.add_argument("--robot_rot", type=float, default=180.0, help="Rotation of the robot around Z axis in degrees.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# imports after Isaac Sim is running
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics
import isaaclab.sim as sim_utils

def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0", dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Spawn custom brown ground plane as a large thin cuboid
    #   Top face lies at Z = 0.0
    sim_utils.spawn_cuboid(
        prim_path="/World/GroundPlane",
        cfg=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.01),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.23, 0.14)), # Brown color
        )
    )
    stage = omni.usd.get_context().get_stage()
    xform_ground = UsdGeom.Xformable(stage.GetPrimAtPath("/World/GroundPlane"))
    xform_ground.ClearXformOpOrder()
    xform_ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/GroundPlane"))

    # Spawn dome light
    sim_utils.spawn_light(prim_path="/World/Light", cfg=sim_utils.DomeLightCfg(intensity=3000.0))

    # Convert rotation angle to quaternion (w, x, y, z) for Z-axis rotation
    import math
    half_angle = math.radians(args_cli.robot_rot) / 2.0
    w = math.cos(half_angle)
    z = math.sin(half_angle)
    orientation = (w, 0.0, 0.0, z)

    # Spawn SO-101 robot at the origin (0.0, 0.0, 0.0)
    robot_cfg = sim_utils.UsdFileCfg(
        usd_path="/home/zozo/leisaac/assets/robots/so101_follower.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,  # Lock base in place
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=16,
        )
    )
    sim_utils.spawn_from_usd(
        prim_path="/World/Robot",
        cfg=robot_cfg,
        translation=(0.0, 0.0, 0.0), # Sitting directly on the ground plane (Z = 0.0)
        orientation=orientation
    )

    # Spawn red cube asset - swapped to the right/negative-X side of the workspace (X = -0.05)
    cube_cfg = sim_utils.UsdFileCfg(
        usd_path="/home/zozo/leisaac/assets/scenes/custom_scene/cube/cube.usd"
    )
    sim_utils.spawn_from_usd(
        prim_path="/World/red_cube",
        cfg=cube_cfg,
        translation=(-0.05, 0.28, 0.0)
    )

    # Spawn grey box asset - swapped to the left/positive-X side of the workspace (X = 0.05)
    box_asset_cfg = sim_utils.UsdFileCfg(
        usd_path="/home/zozo/leisaac/assets/scenes/custom_scene/box/box.usd"
    )
    sim_utils.spawn_from_usd(
        prim_path="/World/grey_box",
        cfg=box_asset_cfg,
        translation=(0.05, 0.28, 0.0)
    )

    # Camera settings configuration (PinholeCameraCfg matching custom task environment templates)
    cam_cfg = sim_utils.PinholeCameraCfg(
        clipping_range=(0.01, 1.0e5),
        focal_length=24.0,
        horizontal_aperture=20.955,
    )

    # Camera 1: setup_cam (now placed on the right side looking left)
    #   Position: X = -32 cm, Y = 28 cm, Z = 25 cm relative to robot base (-0.32, 0.28, 0.25)
    #   Look-at Target: (0.0, 0.28, 0.0) - looking perpendicular across the Y axis, looking at the objects
    sim_utils.spawn_camera(
        prim_path="/World/setup_cam",
        cfg=cam_cfg
    )
    xform_cam1 = UsdGeom.Xformable(stage.GetPrimAtPath("/World/setup_cam"))
    xform_cam1.ClearXformOpOrder()
    # Looks from (-0.32, 0.28, 0.25) to (0.0, 0.28, 0.0) -> perpendicular in XY plane, facing right-to-left (+X direction)
    lookat1 = Gf.Matrix4d().SetLookAt(Gf.Vec3d(-0.32, 0.28, 0.25), Gf.Vec3d(0.0, 0.28, 0.0), Gf.Vec3d(0.0, 0.0, 1.0))
    xform_cam1.AddTransformOp().Set(lookat1.GetInverse())

    # Camera 3: gripper_cam (wrist camera)
    #   Mounted as a child of the gripper link: `/World/Robot/gripper/wrist_camera`
    #   Matching pos, rot, and focal length from leisaac/tasks/template/single_arm_env_cfg.py
    wrist_cam_cfg = sim_utils.PinholeCameraCfg(
        focal_length=36.5,
        focus_distance=400.0,
        horizontal_aperture=36.83,
        clipping_range=(0.01, 50.0),
        lock_camera=True,
    )
    sim_utils.spawn_camera(
        prim_path="/World/Robot/gripper/wrist_camera",
        cfg=wrist_cam_cfg
    )
    wrist_cam_prim = stage.GetPrimAtPath("/World/Robot/gripper/wrist_camera")
    xform_wrist_cam = UsdGeom.Xformable(wrist_cam_prim)
    xform_wrist_cam.ClearXformOpOrder()
    # Converted from ROS convention (offset pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914))
    # to USD/OpenGL convention (w,x,y,z) = (0.912179, -0.404379, 0.0486914, 0.0451242)
    xform_wrist_cam.AddTranslateOp().Set(Gf.Vec3d(-0.001, 0.1, -0.04))
    xform_wrist_cam.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.912179, -0.404379, 0.0486914, 0.0451242))

    # Start simulation play
    sim.reset()
    print("[INFO] Standalone setup environment running! Orbit camera with right-click + drag.")
    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()
