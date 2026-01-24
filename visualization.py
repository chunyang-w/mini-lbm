"""
Visualization module for LBM simulations.
Provides real-time animation and video export without impacting simulation performance.

Author:             Chunyang Wang
Date:               5th Nov 2025
Github username:    chunyang-w
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from typing import Optional, Callable
import threading
import queue
import time


class LBMVisualizer:
    """
    Non-blocking visualizer for LBM simulations.

    Decouples visualization from simulation by using frame buffering.
    This ensures visualization does not impact simulation performance.
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        figsize: tuple = (12, 4),
        cmap: str = 'coolwarm',
        vmin: float = 0.0,
        vmax: float = None,
        title: str = 'LBM Simulation'
    ):
        """
        Initialize the visualizer.

        Parameters
        ----------
        Nx, Ny : int
            Domain dimensions
        figsize : tuple
            Figure size (width, height) in inches
        cmap : str
            Colormap for velocity magnitude
        vmin, vmax : float
            Color scale limits (vmax=None for auto-scaling)
        title : str
            Plot title
        """
        self.Nx = Nx
        self.Ny = Ny
        self.figsize = figsize
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.title = title

        # Frame buffer for saving (stores copies of data)
        self.frames = []
        self.frame_times = []

        # Display state
        self.fig = None
        self.ax = None
        self.im = None
        self.time_text = None
        self.initialized = False

    def _setup_figure(self, initial_data: np.ndarray, obstacle_mask: np.ndarray = None):
        """Set up matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize)

        # Transpose for correct orientation (x horizontal, y vertical)
        display_data = initial_data.T

        # Auto-scale vmax if not set
        if self.vmax is None:
            self.vmax = np.max(initial_data) * 1.2 if np.max(initial_data) > 0 else 0.1

        self.im = self.ax.imshow(
            display_data,
            origin='lower',
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            aspect='equal',
            interpolation='bilinear'
        )

        # Overlay obstacle if provided
        if obstacle_mask is not None:
            obstacle_display = np.ma.masked_where(~obstacle_mask.T, np.ones_like(obstacle_mask.T))
            self.ax.imshow(
                obstacle_display,
                origin='lower',
                cmap='gray',
                alpha=0.8,
                aspect='equal'
            )

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_title(self.title)

        # Colorbar
        cbar = self.fig.colorbar(self.im, ax=self.ax, shrink=0.8)
        cbar.set_label('Velocity magnitude')

        # Time annotation
        self.time_text = self.ax.text(
            0.02, 0.95, '', transform=self.ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        self.fig.tight_layout()
        self.initialized = True

    def update_frame(
        self,
        velocity_magnitude: np.ndarray,
        step: int,
        physical_time: float = None,
        store_frame: bool = True
    ):
        """
        Update visualization with new data.

        Parameters
        ----------
        velocity_magnitude : np.ndarray
            2D array of velocity magnitude
        step : int
            Current simulation step
        physical_time : float, optional
            Physical time in seconds
        store_frame : bool
            Whether to store frame for later saving
        """
        # Store frame for video export (make a copy)
        if store_frame:
            self.frames.append(velocity_magnitude.copy())
            self.frame_times.append((step, physical_time))

        # Update display if initialized
        if self.initialized and self.im is not None:
            self.im.set_data(velocity_magnitude.T)

            if physical_time is not None:
                self.time_text.set_text(f'Step: {step}, t = {physical_time:.4f}s')
            else:
                self.time_text.set_text(f'Step: {step}')

            # Non-blocking draw
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def show(self, velocity_magnitude: np.ndarray, obstacle_mask: np.ndarray = None):
        """
        Initialize and show the live visualization window.

        Parameters
        ----------
        velocity_magnitude : np.ndarray
            Initial velocity magnitude field
        obstacle_mask : np.ndarray, optional
            Boolean mask of obstacle cells
        """
        # Use non-blocking backend if available
        plt.ion()

        self._setup_figure(velocity_magnitude, obstacle_mask)
        plt.show(block=False)
        plt.pause(0.01)

    def close(self):
        """Close the visualization window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.initialized = False

    def save_animation(
        self,
        filename: str,
        fps: int = 30,
        dpi: int = 150,
        obstacle_mask: np.ndarray = None
    ):
        """
        Save buffered frames as animation file.

        Parameters
        ----------
        filename : str
            Output filename (supports .gif, .mp4)
        fps : int
            Frames per second
        dpi : int
            Resolution (dots per inch)
        obstacle_mask : np.ndarray, optional
            Boolean mask of obstacle cells for overlay
        """
        if not self.frames:
            print("No frames to save.")
            return

        print(f"Saving animation with {len(self.frames)} frames to {filename}...")

        # Create new figure for saving (don't interfere with display)
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Get global min/max for consistent color scale
        all_data = np.array(self.frames)
        vmax = np.max(all_data) if self.vmax is None else self.vmax

        # Initial frame
        im = ax.imshow(
            self.frames[0].T,
            origin='lower',
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=vmax,
            aspect='equal',
            interpolation='bilinear'
        )

        # Overlay obstacle
        if obstacle_mask is not None:
            obstacle_display = np.ma.masked_where(~obstacle_mask.T, np.ones_like(obstacle_mask.T))
            ax.imshow(
                obstacle_display,
                origin='lower',
                cmap='gray',
                alpha=0.8,
                aspect='equal'
            )

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(self.title)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Velocity magnitude')

        time_text = ax.text(
            0.02, 0.95, '', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        fig.tight_layout()

        def animate(frame_idx):
            im.set_data(self.frames[frame_idx].T)
            step, phys_time = self.frame_times[frame_idx]
            if phys_time is not None:
                time_text.set_text(f'Step: {step}, t = {phys_time:.4f}s')
            else:
                time_text.set_text(f'Step: {step}')
            return [im, time_text]

        anim = FuncAnimation(
            fig, animate,
            frames=len(self.frames),
            interval=1000/fps,
            blit=True
        )

        # Choose writer based on file extension
        if filename.endswith('.gif'):
            writer = PillowWriter(fps=fps)
        elif filename.endswith('.mp4'):
            try:
                writer = FFMpegWriter(fps=fps, metadata={'title': self.title})
            except Exception:
                print("FFmpeg not available, falling back to GIF format")
                filename = filename.replace('.mp4', '.gif')
                writer = PillowWriter(fps=fps)
        else:
            # Default to GIF
            filename = filename + '.gif'
            writer = PillowWriter(fps=fps)

        anim.save(filename, writer=writer, dpi=dpi)
        plt.close(fig)

        print(f"Animation saved to {filename}")

    def clear_frames(self):
        """Clear stored frames to free memory."""
        self.frames = []
        self.frame_times = []


def create_visualizer_callback(
    visualizer: LBMVisualizer,
    solver,
    dt: float = None,
    show_live: bool = True
) -> Callable:
    """
    Create a callback function for use with LBM2D.run().

    Parameters
    ----------
    visualizer : LBMVisualizer
        The visualizer instance
    solver : LBM2D
        The LBM solver instance
    dt : float, optional
        Physical time step for time display
    show_live : bool
        Whether to show live visualization

    Returns
    -------
    callback : Callable
        Callback function compatible with LBM2D.run()
    """
    # Initialize display on first call
    first_call = [True]

    def callback(step: int, solver):
        vel_mag = solver.get_velocity_magnitude()
        phys_time = step * dt if dt is not None else None

        if first_call[0] and show_live:
            visualizer.show(vel_mag, solver.obstacle)
            first_call[0] = False

        visualizer.update_frame(vel_mag, step, phys_time, store_frame=True)

        if show_live:
            plt.pause(0.001)  # Brief pause for display update

    return callback
