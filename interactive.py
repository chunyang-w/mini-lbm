"""
Interactive obstacle drawing for LBM simulations.

Allows users to draw and erase obstacles using mouse interaction
while the simulation is running.

Controls:
    Left click + drag:  Draw obstacles
    Right click + drag: Erase obstacles
    Scroll wheel / +/-: Change brush size
    Space:              Pause/resume simulation
    c:                  Clear all obstacles
    q:                  Quit

Author:             Chunyang Wang
Date:               2025
Github username:    chunyang-w
"""
import numpy as np


class ObstacleDrawer:
    """
    Mouse-based obstacle drawing on the LBM simulation domain.

    Connects to matplotlib event system for interactive drawing.
    Pending obstacle changes are batched and applied via flush()
    to avoid modifying solver state during a time step.
    """

    def __init__(self, fig, ax, solver, brush_radius=3):
        """
        Initialize the obstacle drawer.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to attach events to
        ax : matplotlib.axes.Axes
            The axes containing the simulation image
        solver : LBM2D
            The LBM solver instance
        brush_radius : int
            Initial brush radius in lattice units
        """
        self.fig = fig
        self.ax = ax
        self.solver = solver
        self.brush_radius = brush_radius
        self.min_brush = 1
        self.max_brush = 20

        # Interaction state
        self.drawing = False
        self.erasing = False
        self.paused = False
        self.should_quit = False

        # Pending cell changes (batched for performance)
        self._pending_add = set()
        self._pending_remove = set()
        self._dirty = False

        # RGBA obstacle overlay (managed separately from visualizer)
        self._obstacle_im = None
        self._init_overlay()

        # Status text (top-right corner)
        self._status_text = ax.text(
            0.98, 0.95, '',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            zorder=10
        )
        self._update_status()

        # Connect matplotlib events
        self._cids = []
        self._cids.append(fig.canvas.mpl_connect('button_press_event', self._on_press))
        self._cids.append(fig.canvas.mpl_connect('button_release_event', self._on_release))
        self._cids.append(fig.canvas.mpl_connect('motion_notify_event', self._on_motion))
        self._cids.append(fig.canvas.mpl_connect('key_press_event', self._on_key))
        self._cids.append(fig.canvas.mpl_connect('scroll_event', self._on_scroll))
        self._cids.append(fig.canvas.mpl_connect('close_event', self._on_close))

    def _init_overlay(self):
        """Create the RGBA obstacle overlay image."""
        # Transparent RGBA overlay (Ny rows x Nx cols to match transposed display)
        overlay = np.zeros((self.solver.Ny, self.solver.Nx, 4))
        self._obstacle_im = self.ax.imshow(
            overlay,
            origin='lower',
            aspect='equal',
            zorder=5
        )

    def _update_status(self):
        """Update the on-screen status text."""
        state = "PAUSED" if self.paused else "RUNNING"
        self._status_text.set_text(
            f'Brush: {self.brush_radius} | {state}\n'
            f'LMB: draw  RMB: erase\n'
            f'Space: pause  c: clear  q: quit'
        )

    # ---- Event handlers ----

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Left click — draw
            self.drawing = True
            self._paint(event.xdata, event.ydata, add=True)
        elif event.button == 3:  # Right click — erase
            self.erasing = True
            self._paint(event.xdata, event.ydata, add=False)

    def _on_release(self, event):
        self.drawing = False
        self.erasing = False

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return
        if self.drawing:
            self._paint(event.xdata, event.ydata, add=True)
        elif self.erasing:
            self._paint(event.xdata, event.ydata, add=False)

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            self._update_status()
            self.fig.canvas.draw_idle()
        elif event.key == 'c':
            self._queue_clear_all()
        elif event.key == 'q':
            self.should_quit = True
        elif event.key in ('+', '='):
            self.brush_radius = min(self.brush_radius + 1, self.max_brush)
            self._update_status()
            self.fig.canvas.draw_idle()
        elif event.key == '-':
            self.brush_radius = max(self.brush_radius - 1, self.min_brush)
            self._update_status()
            self.fig.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.button == 'up':
            self.brush_radius = min(self.brush_radius + 1, self.max_brush)
        elif event.button == 'down':
            self.brush_radius = max(self.brush_radius - 1, self.min_brush)
        self._update_status()
        self.fig.canvas.draw_idle()

    def _on_close(self, event):
        self.should_quit = True

    # ---- Drawing logic ----

    def _paint(self, x, y, add=True):
        """
        Queue obstacle cells in a circular brush area.

        The display uses data.T with origin='lower', so
        xdata maps to lattice x and ydata maps to lattice y.
        """
        if x is None or y is None:
            return

        ci = int(round(x))
        cj = int(round(y))
        r = self.brush_radius
        Nx, Ny = self.solver.Nx, self.solver.Ny

        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if di * di + dj * dj <= r * r:
                    ni, nj = ci + di, cj + dj
                    # Protect domain boundaries:
                    # inlet (i=0), outlet (i=Nx-1), walls (j=0, j=Ny-1)
                    if 1 <= ni <= Nx - 2 and 1 <= nj <= Ny - 2:
                        if add:
                            self._pending_add.add((ni, nj))
                            self._pending_remove.discard((ni, nj))
                        else:
                            self._pending_remove.add((ni, nj))
                            self._pending_add.discard((ni, nj))
        self._dirty = True

    def _queue_clear_all(self):
        """Queue removal of all existing obstacles."""
        obstacle = self.solver.obstacle
        indices = np.where(obstacle)
        for i, j in zip(indices[0], indices[1]):
            self._pending_remove.add((i, j))
        self._pending_add.clear()
        self._dirty = True

    # ---- Public interface ----

    def flush(self):
        """
        Apply pending obstacle changes to the solver.

        Call this from the simulation loop between time steps.
        Returns True if any changes were applied.
        """
        if not self._dirty:
            return False

        changed = False

        if self._pending_add:
            cells = list(self._pending_add)
            self.solver.set_obstacle_cells(cells)
            self._pending_add.clear()
            changed = True

        if self._pending_remove:
            cells = list(self._pending_remove)
            self.solver.clear_obstacle_cells(cells)
            self._pending_remove.clear()
            changed = True

        self._dirty = False
        return changed

    def update_obstacle_overlay(self):
        """Refresh the obstacle overlay image from the solver's obstacle mask."""
        overlay = np.zeros((self.solver.Ny, self.solver.Nx, 4))
        obstacle_t = self.solver.obstacle.T  # Transpose to match display
        overlay[obstacle_t, :] = [0.3, 0.3, 0.3, 0.8]  # Dark gray, semi-transparent
        self._obstacle_im.set_data(overlay)

    def disconnect(self):
        """Disconnect all matplotlib event handlers."""
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)
        self._cids.clear()
