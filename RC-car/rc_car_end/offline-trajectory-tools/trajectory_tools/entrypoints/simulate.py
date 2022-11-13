from trajectory_tools.simulator.model.vehicle import Vehicle
from trajectory_tools.utils.utils import load_ttl, save_ttl
from trajectory_tools.simulator.simulator import Simulator
import numpy as np
import yaml
import os


def main():
    config_path = os.path.join(os.getcwd(), "simulation_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            "Cannot find simulation_config.yaml in the current directory. Execute trajectory_init to create a workspace."
        )
    with open(config_path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)

    ttl_path = config["simulation"]["input_ttl"]
    assert os.path.exists(ttl_path), f"TTL {ttl_path} does not exist."

    traj = load_ttl(ttl_path)

    vehicle = Vehicle(
        downforce_speed_lookup=np.array([[], []]),
        steer_radius_speed_lookup=np.array([[], []]),
        acc_speed_lookup=np.array(config["vehicle"]["acc_speed_lookup"], dtype=float).T,
        dcc_speed_lookup=np.array(config["vehicle"]["dcc_speed_lookup"], dtype=float).T,
        g_circle_radius_mpss=float(config["vehicle"]["g_circle_radius_mpss"]),
    )

    simulator = Simulator(vehicle)
    result = simulator.run_simulation(traj, enable_vis=True)
    print(result)
    output_path = config["simulation"]["output_ttl"]
    print(f"Saving ttl to {output_path}")
    save_ttl(output_path, result.trajectory)


if __name__ == "__main__":
    main()
