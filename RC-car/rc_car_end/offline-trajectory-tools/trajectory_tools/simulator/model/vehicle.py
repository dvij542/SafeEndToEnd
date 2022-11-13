import numpy as np
from dataclasses import dataclass

GRAVITY = 9.81


@dataclass
class Vehicle:
    downforce_speed_lookup: np.ndarray
    steer_radius_speed_lookup: np.ndarray
    acc_speed_lookup: np.ndarray
    dcc_speed_lookup: np.ndarray
    mass_kg: float = 226.0
    wheel_base: float = 1.05
    track: float = 0.95
    front_load_ratio: float = 0.45
    max_steer_rad: float = 0.523599
    max_speed_mps: float = 30.0
    max_jerk: float = 100.0
    g_circle_radius_mpss: float = 15.0
    x_front: float = 1.0
    x_rear: float = -0.2
    y_left: float = 0.5
    y_right: float = 0.5

    def lookup_downforce_from_speed(self, speed_mps: float):
        return np.interp(
            speed_mps,
            self.downforce_speed_lookup[0, :],
            self.downforce_speed_lookup[1, :],
        )

    def lookup_steer_radius_from_speed(self, speed_mps: float):
        return np.interp(
            speed_mps,
            self.steer_radius_speed_lookup[0, :],
            self.steer_radius_speed_lookup[1, :],
        )

    def lookup_speed_from_steer_radius(self, steer_radius_m: float):
        return np.interp(
            steer_radius_m,
            np.flip(self.steer_radius_speed_lookup[1, :]),
            np.flip(self.steer_radius_speed_lookup[0, :]),
        )

    def lookup_acc_from_speed(self, speed_mps: float):
        return np.interp(
            speed_mps, self.acc_speed_lookup[0, :], self.acc_speed_lookup[1, :]
        )

    def lookup_dcc_from_speed(self, speed_mps: float):
        return np.interp(
            speed_mps, self.dcc_speed_lookup[0, :], self.dcc_speed_lookup[1, :]
        )

    def lookup_acc_circle(self, lat=None, lon=None):
        if lat is not None and lon is not None:
            return lat**2 + lon**2 <= self.g_circle_radius_mpss**2

        val = None
        if lat is not None:
            val = lat
        elif lon is not None:
            val = lon
        else:
            return None

        if abs(val) > self.g_circle_radius_mpss:
            return 0.0
        else:
            return np.sqrt(self.g_circle_radius_mpss**2 - val**2)


@dataclass
class ModelVehicle:
    downforce_speed_lookup: np.ndarray
    drag_speed_lookup: np.ndarray
    mass: float = 300.0
    wheel_base: float = 1.0
    track: float = 1.0
    wheel_radius: float = 0.15
    tire_friction_coeff: float = 1.0
    cg_distance_to_rear_axle: float = 0.6
    cg_height_above_ground: float = 0.35
    max_steer_rad: float = 0.523599
    max_speed_mps: float = 30.0
    max_jerk: float = 5.0
    x_front: float = 1.0
    x_rear: float = -0.2
    y_left: float = 0.5
    y_right: float = 0.5


@dataclass
class SteadyState:
    load_front: float
    load_rear: float


class VehicleModel:
    def __init__(self, vehicle: ModelVehicle) -> None:
        self.v = vehicle
        front_load = (
            GRAVITY
            * self.v.mass
            * (self.v.wheel_base - self.v.cg_distance_to_rear_axle)
            / self.v.wheel_base
        )
        rear_load = (
            GRAVITY
            * self.v.mass
            * (self.v.cg_distance_to_rear_axle)
            / self.v.wheel_base
        )
        self.ss = SteadyState(front_load, rear_load)

    def get_lon_loads(self, acc: float):
        pass

    def get_min_steer_radius(self, speed: float, acc: float):
        # calculate the load on front and rear axle
        pass
