import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate bipedal robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from dyros_assets.robots.tocabi import Tocabi_CFG  # isort:skip

joint_names = [ "L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
                "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
                "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Elbow_Joint", "L_Armlink_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
                "Neck_Joint", "Head_Joint",
                "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Elbow_Joint", "R_Armlink_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"]

def design_scene(sim: sim_utils.SimulationContext) -> tuple[list, torch.Tensor]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define origins
    origins = torch.tensor([
        [0.0, 0.0, 0.0],
    ]).to(device=sim.device)

    # Robots
    tocabi = Articulation(Tocabi_CFG.replace(prim_path="/World/Tocabi"))
    robots = [tocabi]

    return robots, origins


def run_simulator(sim: sim_utils.SimulationContext, robots: list[Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    joint_ids = [robots[0].find_joints(joint_name)[0][0] for joint_name in joint_names]

    # Define gains as tensors
    p_gains = torch.tensor([2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0, 
                            2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0, 
                            6000.0, 10000.0, 10000.0, 
                            400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0, 
                            100.0, 100.0, 
                            400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0], device=sim.device)
    d_gains = torch.tensor([15.0, 50.0, 20.0, 25.0, 24.0, 24.0, 
                            15.0, 50.0, 20.0, 25.0, 24.0, 24.0, 
                            200.0, 100.0, 100.0, 
                            10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0, 
                            3.0, 3.0, 
                            10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0], device=sim.device)

    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            for index, robot in enumerate(robots):
                # reset dof state
                joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        for robot in robots:
            target_effort = p_gains * (robot.data.default_joint_pos[:, joint_ids] - robot.data.joint_pos[:, joint_ids]) - d_gains * robot.data.joint_vel[:, joint_ids]
            robot.set_joint_effort_target(target_effort, joint_ids=joint_ids)
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in robots:
            robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])

    # design scene
    robots, origins = design_scene(sim)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, robots, origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
