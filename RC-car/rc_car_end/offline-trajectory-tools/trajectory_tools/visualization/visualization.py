from trajectory_tools.simulator.model.trajectory import Trajectory
import matplotlib.pyplot as plt


class SimulatorVisualization:
    def __init__(self, trajectory: Trajectory) -> None:
        self.trajectory = trajectory
        plt.ion()

        self.figure, self.axs = plt.subplots(1, 3, figsize=(20, 8))
        self.figure.suptitle("Offline Trajectory Optimization")

        self.axs[0].set_title("Speed (m/s)")
        self.scat_speed = self.axs[0].scatter(
            self.trajectory[:, Trajectory.X],
            self.trajectory[:, Trajectory.Y],
            c=self.trajectory[:, Trajectory.SPEED],
            cmap="plasma",
        )
        self.figure.colorbar(self.scat_speed, ax=self.axs[0])
        self.axs[0].axis("equal")

        self.axs[1].set_title("Lateral Acceleration (m/s^2)")
        self.scat_lat_acc = self.axs[1].scatter(
            self.trajectory[:, Trajectory.X],
            self.trajectory[:, Trajectory.Y],
            c=self.trajectory[:, Trajectory.LAT_ACC],
            cmap="plasma",
        )
        self.figure.colorbar(self.scat_lat_acc, ax=self.axs[1])
        self.axs[1].axis("equal")

        self.axs[2].set_title("Longitudinal Acceleration (m/s^2)")
        self.scat_lon_acc = self.axs[2].scatter(
            self.trajectory[:, Trajectory.X],
            self.trajectory[:, Trajectory.Y],
            c=self.trajectory[:, Trajectory.LON_ACC],
            cmap="bwr",
        )
        self.figure.colorbar(self.scat_lon_acc, ax=self.axs[2])
        self.axs[2].axis("equal")

    def update_plot(self, sleep_time=0.0):
        self.scat_speed.set_offsets(self.trajectory[:, 0:2])
        self.scat_speed.set_array(self.trajectory[:, Trajectory.SPEED])
        self.scat_speed.autoscale()

        self.scat_lat_acc.set_offsets(self.trajectory[:, 0:2])
        self.scat_lat_acc.set_array(self.trajectory[:, Trajectory.LAT_ACC])
        self.scat_lat_acc.autoscale()

        self.scat_lon_acc.set_offsets(self.trajectory[:, 0:2])
        self.scat_lon_acc.set_array(self.trajectory[:, Trajectory.LON_ACC])
        self.scat_lon_acc.autoscale()

        self.figure.canvas.draw_idle()

        plt.pause(sleep_time)

    def latch_plot(self):
        plt.ioff()
        plt.show()
        plt.ion()
