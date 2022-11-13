from math import inf

import numpy as np

from matplotlib.backend_bases import MouseButton
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Button, TextBox, CheckButtons
import matplotlib.pyplot as plt

import yaml

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from tkinter.messagebox import showinfo

from trajectory_tools.simulator.model.trajectory import (
    BezierTrajectory,
    Bound,
    Region,
    Trajectory,
)
from trajectory_tools.utils.utils import save_ttl


def main():
    fig, ax = plt.subplots()
    ax.set_title("Offline Trajectory Optimization")
    ax.axis("equal")
    pathdata = []
    references = []
    regions = []
    bounds = []

    def open_file(event):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        if file_path:
            path_points = np.loadtxt(
                file_path, dtype=float, delimiter=",", skiprows=0)
            assert (
                len(path_points) % 4 == 0 and len(path_points) > 0
            ), "The file does not contain valid trajectory."
            for i in range(len(path_points) // 4):
                if i == 0:
                    pathdata.append((Path.MOVETO, tuple(path_points[i])))
                else:
                    pathdata.append((Path.LINETO, tuple(path_points[i * 4])))
                pathdata.append((Path.CURVE4, tuple(path_points[i * 4 + 1])))
                pathdata.append((Path.CURVE4, tuple(path_points[i * 4 + 2])))
                pathdata.append((Path.CURVE4, tuple(path_points[i * 4 + 3])))
            plt.close()

    def open_reference(event):
        root = tk.Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames()
        root.destroy()
        for file_path in file_paths:
            references.append(
                np.loadtxt(file_path, dtype=float,
                           delimiter=",", skiprows=1)[:, 0:2]
            )
            ax.plot(references[-1][:, 0], references[-1][:, 1])

    def open_region_file(event):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        if file_path:
            with open(file_path, "r") as f:
                polygon_dict = yaml.load(f, yaml.SafeLoader)
            for _, polygon in polygon_dict.items():
                assert type(polygon) is dict
                vertices = np.loadtxt(
                    polygon['file'], dtype=float, delimiter=',', skiprows=1)[:, 0:2]
                regions.append(
                    Region(
                        polygon["name"],
                        polygon["code"],
                        vertices
                    )
                )
            for region in regions:
                ax.fill(region.vertices[:, 0],
                        region.vertices[:, 1], alpha=0.2)

    def open_boundary_file(event):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        if file_path:
            with open(file_path, "r") as f:
                bound_dict = yaml.load(f, yaml.SafeLoader)
            for _, bound in bound_dict.items():
                assert type(bound) is dict
                vertices = np.loadtxt(
                    bound['file'], dtype=float, delimiter=',', skiprows=1)[:, 0:2]

                bounds.append(
                    Bound(
                        bound["name"],
                        bound["type"],
                        vertices
                    )
                )
            for bound in bounds:
                ax.plot(bound.vertices[:, 0], bound.vertices[:, 1])

    axbopenref = plt.axes([0.3, 0.75, 0.4, 0.1])
    axbopenbound = plt.axes([0.3, 0.6, 0.4, 0.1])
    axbopenregion = plt.axes([0.3, 0.45, 0.4, 0.1])
    axbnew = plt.axes([0.3, 0.3, 0.2, 0.1])
    axbox = fig.add_axes([0.7, 0.3, 0.1, 0.1])
    axbopen = plt.axes([0.3, 0.15, 0.2, 0.1])
    text_box = TextBox(axbox, label="# new nodes")

    def create_new(event):
        if not text_box.text:
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning(
                "Create New Nodes", "Please enter the number of nodes."
            )
            root.destroy()
            return
        num_pt = int(text_box.text)
        rad_inc = 2 * np.pi / num_pt
        rad = 0.0
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        center = np.array([np.average(x_lim), np.average(y_lim)])
        radius = np.min([x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]]) / 2
        for i in range(num_pt):
            if i == num_pt - 1:
                rad = -1.0 * rad_inc
            this_rad = rad
            next_rad = rad + rad_inc
            pt1 = (
                center[0] + np.cos(this_rad) * radius,
                center[1] + np.sin(this_rad) * radius,
            )
            if i == 0:
                pathdata.append((Path.MOVETO, pt1))
            else:
                pathdata.append((Path.LINETO, pt1))
            ct1 = (
                pt1[0] + np.cos(this_rad + np.pi / 2) * radius / 4,
                pt1[1] + np.sin(this_rad + np.pi / 2) * radius / 4,
            )
            pathdata.append((Path.CURVE4, ct1))
            pt2 = (
                center[0] + np.cos(next_rad) * radius,
                center[1] + np.sin(next_rad) * radius,
            )
            ct2 = (
                pt2[0] + np.cos(next_rad - np.pi / 2) * radius / 4,
                pt2[1] + np.sin(next_rad - np.pi / 2) * radius / 4,
            )
            pathdata.append((Path.CURVE4, ct2))
            pathdata.append((Path.CURVE4, pt2))
            rad += rad_inc
        plt.close()

    bopen = Button(axbopen, "Open Saved")
    bopen.on_clicked(open_file)
    bnew = Button(axbnew, "New")
    bnew.on_clicked(create_new)
    bopenref = Button(axbopenref, "Display Reference TTL")
    bopenref.on_clicked(open_reference)
    bopenregion = Button(axbopenregion, "Display Reference Regions")
    bopenregion.on_clicked(open_region_file)
    bopenbound = Button(axbopenbound, "Display Reference Boundaries")
    bopenbound.on_clicked(open_boundary_file)
    plt.show()

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.1)

    if len(pathdata) == 0:
        print("No trajectory loaded or created.")
        exit()

    START_POINT = 0
    CONTROL_POINT_1 = 1
    CONTROL_POINT_2 = 2
    END_POINT = 3

    codes, verts = zip(*pathdata)
    path = Path(verts, codes)
    patch = PathPatch(path, facecolor="None", edgecolor="black", alpha=1.0)
    ax.add_patch(patch)

    class PathInteractor:
        """
        An path editor.

        Press 't' to toggle vertex markers on and off.  When vertex markers are on,
        they can be dragged with the mouse.
        """

        showverts = True
        epsilon = 10  # max pixel distance to count as a vertex hit

        def __init__(self, pathpatch):

            self.ax = pathpatch.axes
            canvas = self.ax.figure.canvas
            self.pathpatch = pathpatch
            self.pathpatch.set_animated(True)

            x, y = zip(*self.pathpatch.get_path().vertices)

            x = np.roll(x, 2)
            y = np.roll(y, 2)

            self.lines = []
            for i in range(0, len(x), 4):
                self.lines.append(
                    ax.plot(
                        x[i: i + 4],
                        y[i: i + 4],
                        marker="o",
                        markerfacecolor="r",
                        animated=True,
                    )[0]
                )

            self._ind = None  # the active vertex

            self._lock_heading = False

            canvas.mpl_connect("draw_event", self.on_draw)
            canvas.mpl_connect("button_press_event", self.on_button_press)
            canvas.mpl_connect("key_press_event", self.on_key_press)
            canvas.mpl_connect("button_release_event", self.on_button_release)
            canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
            self.canvas = canvas

        def get_ind_under_point(self, event):
            """
            Return the index of the point closest to the event position or *None*
            if no point is within ``self.epsilon`` to the event position.
            """
            # display coords
            xy = np.asarray(self.pathpatch.get_path().vertices)
            if not len(xy):
                return None
            xyt = self.pathpatch.get_transform().transform(xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
            ind = d.argmin()

            if d[ind] >= self.epsilon:
                ind = None

            return ind

        def on_draw(self, event):
            """Callback for draws."""
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
            self.ax.draw_artist(self.pathpatch)
            for line in self.lines:
                self.ax.draw_artist(line)
            self.canvas.blit(self.ax.bbox)

        def on_button_press(self, event):
            """Callback for mouse button presses."""
            if (
                event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts
            ):
                return
            self._ind = self.get_ind_under_point(event)

        def on_button_release(self, event):
            """Callback for mouse button releases."""
            if event.button != MouseButton.LEFT or not self.showverts:
                return
            self._ind = None

        def on_key_press(self, event):
            """Callback for key presses."""
            if not event.inaxes:
                return
            if event.key == "t":
                self.showverts = not self.showverts
                for line in self.lines:
                    line.set_visible(self.showverts)
                if not self.showverts:
                    self._ind = None
            if event.key == "e":
                if self._lock_heading:
                    print("Unlock heading")
                else:
                    print("Lock heading")
                self._lock_heading = not self._lock_heading
            self.canvas.draw()

        def on_mouse_move(self, event):
            """Callback for mouse movements."""
            if (
                self._ind is None
                or event.inaxes is None
                or event.button != MouseButton.LEFT
                or not self.showverts
            ):
                return

            vertices = self.pathpatch.get_path().vertices
            roll = 2 - 4 * ((self._ind + 2) // 4)
            vertices_roll = np.roll(vertices, roll, axis=0)
            point_type = self._ind % 4
            if point_type == START_POINT or point_type == END_POINT:
                offset = np.array([event.xdata, event.ydata]
                                  ) - vertices[self._ind]
                vertices_roll[0:4] += offset
            elif self._lock_heading:
                if point_type == CONTROL_POINT_1:
                    px, py = vertices_roll[0]
                    qx, qy = vertices_roll[1]
                elif point_type == CONTROL_POINT_2:
                    px, py = vertices_roll[2]
                    qx, qy = vertices_roll[3]
                slope = (py - qy) / (px - qx)
                intercept = py - px * slope
                slope_otho = -1.0 / slope
                intercept_otho = event.ydata - event.xdata * slope_otho
                if abs(slope_otho) == inf:
                    x_intersect, y_intersect = event.xdata, py
                elif slope_otho == 0.0:
                    x_intersect, y_intersect = px, event.ydata
                else:
                    x_intersect = (intercept_otho - intercept) / \
                        (slope - slope_otho)
                    y_intersect = slope * x_intersect + intercept
                vertices_roll[(point_type + 2) % 4] = x_intersect, y_intersect
            else:
                if point_type == CONTROL_POINT_1:
                    vertices_roll[3] = event.xdata, event.ydata
                    px, py = vertices_roll[2]
                    qx, qy = vertices_roll[3]
                    direction = np.arctan2(py - qy, px - qx)
                    d = np.linalg.norm(vertices_roll[1] - vertices_roll[0])
                    vertices_roll[0] = (
                        px + np.cos(direction) * d,
                        py + np.sin(direction) * d,
                    )
                elif point_type == CONTROL_POINT_2:
                    vertices_roll[0] = event.xdata, event.ydata
                    px, py = vertices_roll[1]
                    qx, qy = vertices_roll[0]
                    direction = np.arctan2(py - qy, px - qx)
                    d = np.linalg.norm(vertices_roll[2] - vertices_roll[3])
                    vertices_roll[3] = (
                        px + np.cos(direction) * d,
                        py + np.sin(direction) * d,
                    )
            vertices = np.roll(vertices_roll, -1 * roll, axis=0)
            self.pathpatch.get_path().vertices = vertices
            vertices_roll = np.roll(vertices, 2, axis=0)

            for i in range(len(self.lines)):
                self.lines[i].set_data(vertices_roll[i * 4: (i + 1) * 4].T)

            self.canvas.restore_region(self.background)
            self.ax.draw_artist(self.pathpatch)
            for line in self.lines:
                self.ax.draw_artist(line)
            self.canvas.blit(self.ax.bbox)

        def to_bezier_trajectory(self) -> BezierTrajectory:
            vertices = self.pathpatch.get_path().vertices
            vertices = np.roll(vertices, 2, axis=0)
            traj = BezierTrajectory(len(vertices) // 4)
            for i in range(len(traj.points)):
                yaw = np.arctan2(
                    vertices[i * 4 + 3, 1] - vertices[i * 4, 1],
                    vertices[i * 4 + 3, 0] - vertices[i * 4, 0],
                )
                fwd = np.linalg.norm(vertices[i * 4 + 3] - vertices[i * 4 + 2])
                bwd = np.linalg.norm(vertices[i * 4] - vertices[i * 4 + 1])
                lim_ax, lim_ay = vertices[i * 4 + 1]  # Same as below
                lim_bx, lim_by = vertices[i * 4 + 2]  # Save as above
                ratio = 0.0
                traj.points[i] = np.array(
                    [yaw, fwd, bwd, lim_ax, lim_ay, lim_bx, lim_by, ratio]
                )
            return traj

    interactor = PathInteractor(patch)
    ax.set_title("Create the Initial Path Curve")
    ax.axis("equal")
    for reference in references:
        ax.plot(reference[:, 0], reference[:, 1], alpha=0.9)
    for region in regions:
        ax.fill(region.vertices[:, 0], region.vertices[:, 1], alpha=0.2)
    for bound in bounds:
        ax.plot(bound.vertices[:, 0], bound.vertices[:, 1])
    def lock_headings(label):
        if label == "Lock Headings":
            if interactor._lock_heading:
                print("Unlock heading")
            else:
                print("Lock heading")
            interactor._lock_heading = not interactor._lock_heading
        if label == "Hide Vertices":
            interactor.showverts = not interactor.showverts
            for line in interactor.lines:
                line.set_visible(interactor.showverts)
            if not interactor.showverts:
                interactor._ind = None
        interactor.canvas.draw()

    axclock = plt.axes([0.1, 0.0, 0.1, 0.05])
    clock = CheckButtons(
        axclock, ["Lock Headings", "Hide Vertices"], [False, False])
    clock.on_clicked(lock_headings)

    def save(event):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            confirmoverwrite=True, defaultextension="csv", initialfile="curve.csv"
        )
        root.destroy()
        np.savetxt(file_path, patch.get_path().vertices, delimiter=",")

    axbsave = plt.axes([0.3, 0.0, 0.1, 0.05])
    bsave = Button(axbsave, "Save")
    bsave.on_clicked(save)

    def export(event):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            confirmoverwrite=True, defaultextension="csv", initialfile="ttl.csv"
        )
        ttl_num = simpledialog.askinteger(
            "TTL Number", "Enter the TTL Number", initialvalue=0, minvalue=0
        )
        interval = simpledialog.askfloat(
            "Waypoint Interval",
            "Enter the waypoint interval (m)",
            initialvalue=0.2,
            minvalue=0.0,
        )
        root.destroy()

        try:
            assert ttl_num is not None
            assert interval is not None and interval > 0.0
            b_traj = interactor.to_bezier_trajectory()
            traj = BezierTrajectory.sample_along(
                b_traj.get_all_curves(), interval, evenly_space=True
            )
            traj.ttl_num = ttl_num
            if len(regions) > 0:
                traj.fill_region(regions)
            if len(bounds) > 0:
                traj.fill_bounds(bounds, 20.0)
            save_ttl(file_path, traj)
            root = tk.Tk()
            root.withdraw()
            showinfo(
                "Export TTL", f"TTL {ttl_num} successfully exported to {file_path}"
            )
            root.destroy()
        except Exception as e:
            root = tk.Tk()
            root.withdraw()
            showinfo(
                "Export TTL", f"TTL {ttl_num} failed to export\nException details:\n{e}"
            )
            root.destroy()

    axbexport = plt.axes([0.5, 0.0, 0.1, 0.05])
    bexport = Button(axbexport, "Export TTL")
    bexport.on_clicked(export)

    plt.show()


if __name__ == "__main__":
    main()
