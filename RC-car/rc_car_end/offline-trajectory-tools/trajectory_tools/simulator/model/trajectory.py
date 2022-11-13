from dataclasses import dataclass
import matplotlib.pyplot as plt
import math
import numpy as np
from bezier.curve import Curve
from shapely.geometry import Point, Polygon, asLinearRing, asLineString, LineString, GeometryCollection


@dataclass
class Region:
    name: str
    code: int
    vertices: np.ndarray  # n * 2

@dataclass
class Bound:
    name: str
    type: str
    vertices: np.ndarray


class Trajectory:
    X = 0
    Y = 1
    Z = 2
    YAW = 3
    SPEED = 4
    CURVATURE = 5
    DIST_TO_SF_BWD = 6
    DIST_TO_SF_FWD = 7
    REGION = 8
    LEFT_BOUND_X = 9
    LEFT_BOUND_Y = 10
    RIGHT_BOUND_X = 11
    RIGHT_BOUND_Y = 12
    LON_ACC = 13
    LAT_ACC = 14
    TIME = 15
    IDX = 16
    ITERATION_FLAG = 17

    def __init__(self, num_point: int, ttl_num: int = 0) -> None:
        self.ttl_num = ttl_num
        self.points = np.zeros((num_point, 18), dtype=np.float64)
        self.points[:, Trajectory.IDX] = np.arange(0, len(self.points), 1)
        self.points[:, Trajectory.ITERATION_FLAG] = -1

    def get_curvature_from_three_points(
        p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
    ) -> float:
        try:
            x, y, z = (
                complex(p1[0], p1[1]),
                complex(p2[0], p2[1]),
                complex(p3[0], p3[1]),
            )
            w = z - x
            w /= y - x
            c = (x - y) * (w - abs(w) ** 2) / 2j / w.imag - x
            return abs(c + x)
        except ZeroDivisionError:
            return math.inf

    def distance(self, pt1, pt2):
        return math.sqrt(
            (pt1[Trajectory.X] - pt2[Trajectory.X]) ** 2
            + (pt1[Trajectory.Y] - pt2[Trajectory.Y]) ** 2
        )

    def fill_curvature(self):
        for i in range(len(self.points)):
            a, b, c = i - 1, i, i + 1
            if a < 0:
                a = len(self.points) - 1
            if c >= len(self.points):
                c = 0
            self.points[
                i, Trajectory.CURVATURE
            ] = Trajectory.get_curvature_from_three_points(
                self.points[a, 0:2], self.points[b, 0:2], self.points[c, 0:2]
            )

    def fill_time(self):
        # Check for zero speeds
        for pt in self.points:
            if pt[Trajectory.SPEED] == 0.0 and pt[Trajectory.LON_ACC == 0.0]:
                raise Exception(
                    "Zero speed and lon_acc encoutered. Cannot fill time.")

        self.points[0, Trajectory.TIME] = 0.0
        for i in range(len(self.points)):
            this, next = i, i + 1
            if next == len(self.points):
                next = 0
            # x = 1/2 * (v_0 + v) * t
            x = self.distance(self.points[this], self.points[next])
            self.points[next, Trajectory.TIME] = x / (
                0.5
                * (
                    self.points[this, Trajectory.SPEED]
                    + self.points[next, Trajectory.SPEED]
                )
            )
            self.points[next, Trajectory.TIME] += self.points[this,
                                                              Trajectory.TIME]

    def fill_distance(self):
        self.points[0, Trajectory.DIST_TO_SF_BWD] = 0.0
        self.points[0, Trajectory.DIST_TO_SF_FWD] = 0.0
        for i in range(len(self.points)):
            this, next = i, i + 1
            if next == len(self.points):
                next = 0
            x = self.distance(self.points[this], self.points[next])
            self.points[next, Trajectory.DIST_TO_SF_BWD] = (
                x + self.points[this, Trajectory.DIST_TO_SF_BWD]
            )

        for i in reversed(range(len(self.points))):
            this, next = i, i + 1
            if next == len(self.points):
                next = 0
            x = self.distance(self.points[this], self.points[next])
            self.points[this, Trajectory.DIST_TO_SF_FWD] = (
                x + self.points[next, Trajectory.DIST_TO_SF_FWD]
            )

    def fill_region(self, regions: list):
        polygons = []
        for region in regions:
            polygons.append((Polygon(region.vertices.tolist()), region.code))

        def p_in_p(row: np.ndarray):
            p = Point([row[Trajectory.X], row[Trajectory.Y]])
            for polygon, code in polygons:
                if polygon.contains(p):
                    row[Trajectory.REGION] = code
                    return

        np.apply_along_axis(p_in_p, 1, self.points)

    def fill_bounds(self, bounds: list, max_distance: float):
        """Fill the left and right bounds. **DO NOT call if yaw is not populated.**

        Args:
            bounds (list): list of Bound object
            max_distance (float): max distance for checking bound intersection

        Raises:
            Exception: if any bound contains invalid geometry type
        """
        geoms = []
        for bound in bounds:
            if bound.type == 'ring':
                geoms.append(asLinearRing(bound.vertices))
            elif bound.type == 'line':
                geoms.append(asLineString(bound.vertices))
            else:
                raise Exception(
                    "Invalid boundary type. It can be ring or line.")

        def calc_bounds(row: np.ndarray):
            pt = (row[Trajectory.X], row[Trajectory.Y])
            ttl_point = Point(pt)
            left_ang = row[Trajectory.YAW] + np.pi / 2.0
            left_pt = (row[Trajectory.X] + max_distance * np.cos(left_ang),
                       row[Trajectory.Y] + max_distance * np.sin(left_ang))

            right_ang = row[Trajectory.YAW] - np.pi / 2.0
            right_pt = (row[Trajectory.X] + max_distance * np.cos(right_ang),
                        row[Trajectory.Y] + max_distance * np.sin(right_ang))

            def find_bound(some_pt):
                min_some_distance = max_distance
                some_line = LineString((pt, some_pt))
                some_bound = Point(some_pt)
                for geom in geoms:
                    some_intersects = some_line.intersection(geom)
                    this_some_distance = min_some_distance
                    this_some_intersection = None
                    if type(some_intersects) is GeometryCollection:
                        distances = []
                        intersections = []
                        for intersection in list(GeometryCollection):
                            if type(intersection) is Point:
                                distances.append(
                                    ttl_point.distance(intersection))
                                intersections.append(intersection)
                            else:
                                print(
                                    f"Issue with boundary at index {row[Trajectory.IDX]}: intersection with {geom.name} is not a Point but {type(intersection)}.")
                        if len(distances) == 0:
                            print(
                                f"Issue with boundary at index {row[Trajectory.IDX]}: no Point intersection found with Geometry of name {geom.name}.")
                        else:
                            min_dist_idx = np.argmin(np.array(distances))
                            this_some_distance = distances[min_dist_idx]
                            this_some_intersection = intersections[min_dist_idx]

                    elif type(some_intersects) is Point:
                        this_some_distance = ttl_point.distance(some_intersects)
                        this_some_intersection = some_intersects

                    if this_some_distance < min_some_distance and this_some_intersection is not None:
                        min_some_distance = this_some_distance
                        some_bound = this_some_intersection

                return min_some_distance, some_bound

            _, left_bound = find_bound(left_pt)
            _, right_bound = find_bound(right_pt)

            row[Trajectory.LEFT_BOUND_X] = left_bound.x
            row[Trajectory.LEFT_BOUND_Y] = left_bound.y
            row[Trajectory.RIGHT_BOUND_X] = right_bound.x
            row[Trajectory.RIGHT_BOUND_Y] = right_bound.y

        np.apply_along_axis(calc_bounds, 1, self.points)

    def set(self, idx: int, field: int, val: float):
        self.points[idx, field] = val

    def get(self, idx: int, field: int):
        return self.points[idx, field]

    def inc(self, idx: int):
        if idx + 1 == len(self.points):
            return 0
        else:
            return idx + 1

    def dec(self, idx: int):
        if idx - 1 < 0:
            return len(self.points) - 1
        else:
            return idx - 1

    def copy(self):
        new_traj = Trajectory(len(self.points))
        new_traj.points = self.points.copy()
        return new_traj

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, val):
        self.points[key] = val

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        for pt in self.points:
            yield pt


class BezierPoint:
    YAW = 0
    FWD = 1
    BWD = 2
    LIM_A_X = 3
    LIM_A_Y = 4
    LIM_B_X = 5
    LIM_B_Y = 6
    RATIO = 7

    def get_control_point(arr: np.ndarray):
        return np.array(
            [
                arr[BezierPoint.LIM_A_X]
                + (arr[BezierPoint.LIM_B_X] - arr[BezierPoint.LIM_A_X])
                * arr[BezierPoint.RATIO],
                arr[BezierPoint.LIM_A_Y]
                + (arr[BezierPoint.LIM_B_Y] - arr[BezierPoint.LIM_A_Y])
                * arr[BezierPoint.RATIO],
            ],
            dtype=np.float64,
        )

    def get_fwd_node(arr: np.ndarray):
        ctrl_pt = BezierPoint.get_control_point(arr)
        return np.array(
            [
                ctrl_pt[0] + arr[BezierPoint.FWD] *
                np.cos(arr[BezierPoint.YAW]),
                ctrl_pt[1] + arr[BezierPoint.FWD] *
                np.sin(arr[BezierPoint.YAW]),
            ]
        )

    def get_bwd_node(arr: np.ndarray):
        ctrl_pt = BezierPoint.get_control_point(arr)
        return np.array(
            [
                ctrl_pt[0]
                + arr[BezierPoint.BWD] * np.cos(arr[BezierPoint.YAW] + np.pi),
                ctrl_pt[1]
                + arr[BezierPoint.BWD] * np.sin(arr[BezierPoint.YAW] + np.pi),
            ]
        )


class BezierTrajectory:
    def __init__(self, num_point: int) -> None:
        self.points = np.zeros((num_point, 8), dtype=np.float64)

    def initialize_control_point(self, idx: int, arr: np.ndarray) -> None:
        self.points[idx] = arr

    def get_all_curves(self) -> list:
        return self.get_curves(0, len(self.points))

    def get_curves(self, start_idx: int, length: int) -> list:
        result = []
        while length > 0:
            result.append(self.get_curve(start_idx))
            start_idx += 1
            length -= 1
            if start_idx >= len(self.points):
                start_idx = 0
        return result

    def get_curve(self, start_idx: int) -> Curve:
        length = 2
        end_idx = start_idx + length
        if end_idx >= len(self.points):
            end_idx -= len(self.points)
        if end_idx == start_idx:
            return None
        elif end_idx < start_idx:
            shift = -(end_idx + 1)
            new_start_idx = start_idx + shift
            new_end_idx = new_start_idx + length
            points = np.roll(self.points, shift=-(end_idx + 1), axis=0)[
                new_start_idx:new_end_idx
            ]
        else:
            points = self.points[start_idx:end_idx]
        return Curve.from_nodes(self.get_nodes(points))

    def get_nodes(self, arr: np.array):
        result = None
        for bezier_pt in arr:
            new = np.column_stack(
                (
                    BezierPoint.get_bwd_node(bezier_pt),
                    BezierPoint.get_control_point(bezier_pt),
                    BezierPoint.get_fwd_node(bezier_pt),
                )
            )
            if result is not None:
                result = np.column_stack((result, new))
            else:
                result = np.copy(new)
        return result[:, 1:-1]

    def sample_along(curves: list, interval: float, evenly_space=False) -> Trajectory:
        """Sample along a trajectory

        Args:
            curves (list): a list of Beizer curves that represent the trajectory.
            interval (float): interval of sample in meter.
            evenly_space (bool, optional): If true, slightly modify the interval to ensure envenly spaced samples. Defaults to False.

        Returns:
            np.ndarray: `(num_sample, 3)` where each row represents the `x`, `y`, and `yaw` of a trajectory point.
        """
        total_length = BezierTrajectory.get_length(curves)
        num_sample = int(total_length // interval)
        if evenly_space:
            interval = total_length / num_sample
        current_curve = 0
        current_length = 0.0
        traj = Trajectory(num_sample)
        for i in range(num_sample):
            while curves[current_curve].length < current_length:
                current_length -= curves[current_curve].length
                current_curve += 1
            percentage = current_length / curves[current_curve].length
            traj[i, 0:2] = (
                curves[current_curve]
                .evaluate(percentage)
                .T
            )
            x, y = curves[current_curve].evaluate_hodograph(percentage).squeeze()
            traj[i, Trajectory.YAW] = np.arctan2(y, x)
            current_length += interval
        traj.fill_curvature()
        traj.fill_distance()
        return traj

    def get_length(curves: list) -> float:
        length = 0.0
        for curve in curves:
            length += curve.length
        return length


if __name__ == "__main__":
    poly = BezierTrajectory(4)
    poly.points = np.array(
        [
            [5.49779, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 0.5],
            [0.785398, 1.0, 1.0, 3.0, -1.0, 5.0, 1.0, 0.5],
            [2.35619, 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 0.5],
            [3.92699, 1.0, 1.0, -1.0, 3.0, 1.0, 5.0, 0.5],
        ]
    )

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    curves = poly.get_curves(0, 4)
    traj = BezierTrajectory.sample_along(curves, 0.1)
    ax.plot(traj[:, Trajectory.X], traj[:, Trajectory.Y])

    control_pts = np.zeros((4, 2))

    for i in range(4):
        control_pts[i] = BezierPoint.get_control_point(poly.points[i])
    ax.scatter(control_pts[:, 0], control_pts[:, 1])

    plt.show()
