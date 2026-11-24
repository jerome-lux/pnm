# -*- coding: utf-8

import os
import vtk
from warnings import warn
import numpy as np
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pnm
from copy import deepcopy


# TODO : change throats_opacitymap and throats_colormap to something else than list...

DEFAULT_COLOR = (1, 1, 1)
DEFAULT_OPACITY = 1

class Fast3DViewer:
    """Fast 3D viewer for pore networks using VTK glyphs and tube filters.

    Unlike :class:`PNM_3d_viewer` (which creates one VTK actor per pore and
    per throat, making it slow and memory-hungry for large networks), this
    class uses a single glyph actor for all pores and a single tube actor for
    all throats. This makes it suitable for networks with tens of thousands
    of pores.

    Parameters
    ----------
    network : pore_network
        The pore network model instance to visualize.
    """

    def __init__(self, network):
        self.network = network
        self.pore_actor = None
        self.throat_actor = None
        # Last render() inputs; reused by save_current_view to rebuild the scene.
        self._pcmap = None
        self._tcmap = None
        self._resolution = 8
        self._periodic = True

    @classmethod
    def fromfile(cls, filename):
        """Create a Fast3DViewer from a saved network file.

        Parameters
        ----------
        filename : str
            Path to the network file (GML/GEXF/pickle format).

        Returns
        -------
        Fast3DViewer
            A new viewer instance containing the loaded network.
        """
        return cls(network=pnm.pore_network.read(filename))

    def reset_renderer(self, bgcolor=(1, 1, 1)):
        """Reset the VTK renderer with a given background color.

        Parameters
        ----------
        bgcolor : tuple of float, optional
            RGB background color (values in [0, 1]). Default is (1, 1, 1)
            (white).
        """
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(bgcolor)
        cam = self.renderer.GetActiveCamera()
        self.camera_props = dict(
            zip(("pos", "focal", "viewup"), (cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp()))
        )

    def create_pores(self, colors=None, resolution=8, **kwargs):
        """Create a single glyph actor representing all pores as spheres.

        Each pore is rendered as a sphere scaled by its radius using
        vtkGlyph3DMapper, which is much faster than creating one actor per pore.

        Parameters
        ----------
        colors : list of tuple or None, optional
            RGB colors for each pore (in node order). If None, all pores use
            the default actor color. Default is None.
        resolution : int, optional
            Sphere mesh resolution (theta and phi). Default is 8.
        **kwargs
            Additional keyword arguments (currently unused).
        """

        print(f"Creating pore glyph actor (resolution={resolution})")

        centers = list(nx.get_node_attributes(self.network.graph, "center").values())
        radii = list(nx.get_node_attributes(self.network.graph, "radius").values())

        self.pore_actor = create_sphere_glyphs(coord=centers, radii=radii, colors=colors)

    def create_throats(self, colors=None, resolution=6, periodic=True, pore_colors=None):
        """Create a single tube actor representing all throats as cylinders.

        Periodic throats are rendered as two tubes each: one crossing the
        box boundary in the positive direction, and one in the negative
        direction, to visualize the periodicity.

        Parameters
        ----------
        colors : list of tuple or None, optional
            RGB colors for each throat (in edge order). If None, all throats
            use the default actor color. Default is None.
        resolution : int, optional
            Tube mesh resolution (number of sides). Default is 6.
        periodic : bool, optional
            Coloring mode for periodic throats. If True (default), both halves
            of a periodic throat share its per-edge color — the current behavior
            (for node-only fields that is the mean of the two endpoint pores).
            If False and ``pore_colors`` is supplied, each half-capillary is
            colored with the color of the real pore it connects to on that side
            of the box. Default is True.
        pore_colors : dict or None, optional
            Mapping from node label to RGB color, used only when ``periodic`` is
            False to color each periodic half-capillary by its same-side pore.
            A missing lookup falls back to the per-edge color (or the default
            color). Default is None.

        Notes
        -----
        The per-endpoint color array must stay aligned with the point array:
        every appended segment contributes exactly two entries. When
        ``periodic`` is False and ``pore_colors`` is given but ``colors`` is not,
        non-periodic throats fall back to the mean of their endpoint pore colors
        so this alignment is preserved.
        """

        heads = []
        tails = []
        radii = []
        cmap = []

        # When ``periodic`` is False and per-pore colors are supplied, each half of a
        # periodic throat is colored with the color of the real pore it connects to on
        # that side of the box (instead of the shared per-edge color).
        use_pore_colors = (not periodic) and pore_colors is not None

        extent = self.network.graph.graph.get("extent")
        if extent is None:
            warn("Network extent is not set. Periodic throats may not be displayed correctly.")

        for i, (n1, n2, props) in enumerate(self.network.graph.edges.data()):
            c1 = np.array(self.network.graph.nodes[n1]["center"])
            c2 = np.array(self.network.graph.nodes[n2]["center"])

            # Trick to visualize periodic throats
            if props.get("periodic", False):
                ind = {"x": 0, "y": 1, "z": 2}
                pdir = props.get("pdir", "x")

                # Real pore on the +pdir / -pdir side of the box.
                if c1[ind[pdir]] > c2[ind[pdir]]:
                    plus_pore, minus_pore = n1, n2
                else:
                    plus_pore, minus_pore = n2, n1

                # Uniform color for each half-capillary (see ``use_pore_colors``).
                if use_pore_colors:
                    fallback = colors[i] if colors is not None else DEFAULT_COLOR
                    color_plus = pore_colors.get(plus_pore, fallback)
                    color_minus = pore_colors.get(minus_pore, fallback)
                elif colors is not None:
                    color_plus = colors[i]
                    color_minus = colors[i]

                # --- Half-capillary crossing the boundary in the +pdir direction. ---
                c1p = deepcopy(c1)
                c2p = deepcopy(c2)
                if c1[ind[pdir]] > c2[ind[pdir]]:
                    if extent is not None:
                        c2p[ind[pdir]] = extent[ind[pdir]] + c2[ind[pdir]]
                        c2p = 0.5 * (c1p + c2p)
                    else:
                        c2p = 0.5 * (c1p + c2p)
                else:
                    if extent is not None:
                        c1p[ind[pdir]] = extent[ind[pdir]] + c1[ind[pdir]]
                        c1p = 0.5 * (c1p + c2p)
                    else:
                        c1p = 0.5 * (c1p + c2p)
                heads.append(c1p)
                tails.append(c2p)
                radii.extend([props.get("radius", 0.1)] * 2)
                if colors is not None or use_pore_colors:
                    cmap.extend([color_plus] * 2)

                # --- Half-capillary crossing the boundary in the -pdir direction. ---
                c1p = deepcopy(c1)
                c2p = deepcopy(c2)
                if c1[ind[pdir]] < c2[ind[pdir]]:
                    if extent is not None:
                        c2p[ind[pdir]] = -extent[ind[pdir]] + c2[ind[pdir]]
                        c2p = 0.5 * (c1p + c2p)
                    else:
                        c2p = 0.5 * (c1p + c2p)
                else:
                    if extent is not None:
                        c1p[ind[pdir]] = -extent[ind[pdir]] + c1[ind[pdir]]
                        c1p = 0.5 * (c1p + c2p)
                    else:
                        c1p = 0.5 * (c1p + c2p)
                heads.append(c1p)
                tails.append(c2p)
                radii.extend([props.get("radius", 0.1)] * 2)
                if colors is not None or use_pore_colors:
                    cmap.extend([color_minus] * 2)

            else:
                heads.append(c1)
                tails.append(c2)
                radii.extend([props.get("radius", 0.1)] * 2)
                if colors is not None:
                    cmap.extend([colors[i]] * 2)
                elif use_pore_colors:
                    # No per-edge color available: fall back to the mean of the two
                    # endpoint pore colors so the color array stays aligned with points.
                    cn1 = pore_colors.get(n1)
                    cn2 = pore_colors.get(n2)
                    if cn1 is not None and cn2 is not None:
                        cmean = tuple(int(round(0.5 * (a + b))) for a, b in zip(cn1, cn2))
                    else:
                        cmean = DEFAULT_COLOR
                    cmap.extend([cmean] * 2)

        self.throat_actor = create_tubes(heads, tails, radii, cmap, resolution=resolution)

    def create_renWin(self, size=(800, 1000)):
        """Create a VTK render window and add the renderer.

        Parameters
        ----------
        size : tuple of int, optional
            Window size (width, height) in pixels. Default is (800, 1000).
        """
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.renWin.SetSize(size)

    def close_renWin(self):
        """Close and delete the render window."""
        if self.renWin is not None:
            self.renWin.Finalize()
            del self.renWin
            self.renWin = None

    def render(self, pcmap=None, tcmap=None, resolution=8, periodic=True):
        """Render the pore network (pores and throats).

        Creates the pore and throat actors if they do not exist yet, then adds
        them to the renderer.

        Parameters
        ----------
        pcmap : list of tuple or None, optional
            Pore colors (RGB tuples), in node order. Default is None.
        tcmap : list of tuple or None, optional
            Throat colors (RGB tuples). Default is None.
        resolution : int, optional
            Mesh resolution for pores and throats. Default is 8.
        periodic : bool, optional
            Periodic-throat coloring mode forwarded to
            :meth:`create_throats`. When False, each half of a periodic throat
            is colored by its same-side pore (requires ``pcmap``). Default is True.
        """

        self.reset_renderer()
        self._pcmap = pcmap
        self._tcmap = tcmap
        self._resolution = resolution
        self._periodic = periodic

        if self.pore_actor is None:
            self.create_pores(colors=pcmap, resolution=resolution)
        self.renderer.AddActor(self.pore_actor)

        if self.throat_actor is None:
            pore_color_map = (
                dict(zip(self.network.graph.nodes(), pcmap))
                if (not periodic and pcmap is not None)
                else None
            )
            self.create_throats(
                colors=tcmap, resolution=resolution, periodic=periodic, pore_colors=pore_color_map
            )
        self.renderer.AddActor(self.throat_actor)

    def interact(self, azimuth=20, elevation=30, size=(800, 1000)):
        """Start an interactive VTK rendering session.

        Parameters
        ----------
        azimuth : float, optional
            Camera azimuth angle in degrees. Default is 20.
        elevation : float, optional
            Camera elevation angle in degrees. Default is 30.
        size : tuple of int, optional
            Window size (width, height). Default is (800, 1000).
        """
        self.create_renWin(size)
        iren = vtk.vtkRenderWindowInteractor()
        self.set_camera(azimuth, elevation)
        self.renWin.OffScreenRenderingOff()
        iren.SetRenderWindow(self.renWin)
        iren.Initialize()

        iren.Start()
        del iren
        # self.close_renWin()

    def reset_camera(self):
        """Restore the camera to its initial position (saved at reset_renderer time)."""
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(self.camera_props["pos"])
        cam.SetFocalPoint(self.camera_props["focal"])
        cam.SetViewUp(self.camera_props["viewup"])

    def set_camera(self, azimuth, elevation):
        """Position the camera with the given azimuth and elevation angles.

        Parameters
        ----------
        azimuth : float
            Camera azimuth angle in degrees.
        elevation : float
            Camera elevation angle in degrees.
        """
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().OrthogonalizeViewUp()
        self.renderer.GetActiveCamera().Azimuth(azimuth)
        self.renderer.GetActiveCamera().Elevation(elevation)
        self.renderer.ResetCameraClippingRange()

    def save(self, filename, pcmap=None, tcmap=None, resolution=8, azimuth=20, elevation=30, size=(800, 1000), periodic=True):
        """Render the network offscreen and save it as a JPEG image.

        Parameters
        ----------
        filename : str
            Output JPEG file path.
        pcmap : list of tuple or None, optional
            Pore colors (RGB tuples), in node order. Default is None.
        tcmap : list of tuple or None, optional
            Throat colors. Default is None.
        resolution : int, optional
            Mesh resolution. Default is 8.
        azimuth : float, optional
            Camera azimuth angle. Default is 20.
        elevation : float, optional
            Camera elevation angle. Default is 30.
        size : tuple of int, optional
            Window size (width, height). Default is (800, 1000).
        periodic : bool, optional
            Periodic-throat coloring mode forwarded to :meth:`render`. When
            False, each half of a periodic throat is colored by its same-side
            pore (requires ``pcmap``). Default is True.

        Notes
        -----
        For an unknown reason, this does not work after interacting with the
        scene (the render window may not close correctly after interaction).
        To export the view adjusted during ``interact()``, use
        :meth:`save_current_view` instead.

        Raises
        ------
        FileNotFoundError
            If the parent directory of ``filename`` does not exist.
        RuntimeError
            If VTK fails to write the image file.
        """
        parent = os.path.dirname(filename)
        if parent and not os.path.isdir(parent):
            raise FileNotFoundError(f"Cannot save image: directory '{parent}' does not exist.")

        self.render(pcmap=pcmap, tcmap=tcmap, resolution=resolution, periodic=periodic)
        self.create_renWin(size)
        self.set_camera(azimuth, elevation)

        self.renWin.OffScreenRenderingOn()
        self.renWin.Render()
        # Create the filter to write the rendered image
        w2i = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkJPEGWriter()
        writer.SetQuality(100)
        w2i.SetInput(self.renWin)
        w2i.Update()
        writer.SetInputConnection(w2i.GetOutputPort())
        # The VTK binding returns None from Write() regardless of the outcome,
        # so detect failure by checking that the file exists and was just updated.
        pre = os.stat(filename) if os.path.exists(filename) else None
        writer.SetFileName(filename)
        writer.Write()
        post = os.stat(filename) if os.path.exists(filename) else None
        if (
            post is None
            or post.st_size == 0
            or (pre is not None and post.st_mtime_ns == pre.st_mtime_ns)
        ):
            raise RuntimeError(f"Failed to write image file '{filename}'.")
        self.close_renWin()

    def save_current_view(self, filename, size=(1600, 2000)):
        """Save an offscreen JPEG of the current camera view.

        Captures the active camera state (position, focal point, view-up and
        projection) from this viewer's renderer — typically after
        ``interact()`` — and re-renders it on a fresh render window without
        resetting the camera, so the exported image matches exactly what was
        seen interactively. Unlike :meth:`save`, no azimuth/elevation is
        applied.

        Parameters
        ----------
        filename : str
            Output JPEG file path.
        size : tuple of int, optional
            Offscreen window size (width, height). Default is (1600, 2000).

        Notes
        -----
        The scene is rebuilt on a fresh viewer instance: ``render()`` resets
        the renderer and camera, and reusing the interacted render window for
        offscreen export is unreliable (see :meth:`save` notes). Requires
        ``render(pcmap=..., tcmap=...)`` to have been called first.

        Raises
        ------
        FileNotFoundError
            If the parent directory of ``filename`` does not exist.
        RuntimeError
            If VTK fails to write the image file.
        """
        parent = os.path.dirname(filename)
        if parent and not os.path.isdir(parent):
            raise FileNotFoundError(f"Cannot save image: directory '{parent}' does not exist.")

        cam = self.renderer.GetActiveCamera()
        state = dict(
            pos=cam.GetPosition(),
            focal=cam.GetFocalPoint(),
            viewup=cam.GetViewUp(),
            parallel=bool(cam.GetParallelProjection()),
            pscale=cam.GetParallelScale(),
        )

        out = Fast3DViewer(self.network)
        out.render(
            pcmap=self._pcmap, tcmap=self._tcmap, resolution=self._resolution, periodic=self._periodic
        )
        out.create_renWin(size=size)

        c2 = out.renderer.GetActiveCamera()
        c2.SetPosition(state["pos"])
        c2.SetFocalPoint(state["focal"])
        c2.SetViewUp(state["viewup"])
        if state["parallel"]:
            c2.SetParallelProjection(1)
            c2.SetParallelScale(state["pscale"])
        out.renderer.ResetCameraClippingRange()

        out.renWin.OffScreenRenderingOn()
        out.renWin.Render()
        w2i = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkJPEGWriter()
        writer.SetQuality(100)
        w2i.SetInput(out.renWin)
        w2i.Update()
        writer.SetInputConnection(w2i.GetOutputPort())
        # The VTK binding returns None from Write() regardless of the outcome,
        # so detect failure by checking that the file exists and was just updated.
        pre = os.stat(filename) if os.path.exists(filename) else None
        writer.SetFileName(filename)
        writer.Write()
        post = os.stat(filename) if os.path.exists(filename) else None
        if (
            post is None
            or post.st_size == 0
            or (pre is not None and post.st_mtime_ns == pre.st_mtime_ns)
        ):
            raise RuntimeError(f"Failed to write image file '{filename}'.")
        out.close_renWin()


class PNM_3d_viewer:
    """Slow 3D viewer for pore networks (one VTK actor per pore and per throat).

    This viewer creates an individual VTK actor for every pore and every throat.
    It is simple and allows independent configuration of each actor (opacity,
    color), but it is slow and memory-hungry for large networks.
    Use :class:`Fast3DViewer` instead for networks with more than a few hundred
    pores.

    Parameters
    ----------
    network : pore_network or None, optional
        The pore network model instance to visualize. Default is None.
    """

    def __init__(self, network=None):
        self.set_network(network)

        self.geom_complete = False

    @classmethod
    def fromfile(cls, filename):
        """Create a PNM_3d_viewer from a saved network file.

        Parameters
        ----------
        filename : str
            Path to the network file (GML/GEXF/pickle format).

        Returns
        -------
        PNM_3d_viewer
            A new viewer instance containing the loaded network.
        """
        return cls(network=pnm.pore_network.read(filename))

    def reset_renderer(self, bgcolor=(1, 1, 1)):
        """Reset the VTK renderer with a given background color.

        Parameters
        ----------
        bgcolor : tuple of float, optional
            RGB background color (values in [0, 1]). Default is (1, 1, 1) (white).
        """
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(bgcolor)
        cam = self.renderer.GetActiveCamera()
        self.camera_props = dict(
            zip(("pos", "focal", "viewup"), (cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp()))
        )

    def set_network(self, network):
        """Set the network to visualize and create an actor graph copy.

        Parameters
        ----------
        network : pore_network
            The pore network model instance.
        """
        self.network = network
        self.actor_graph = self.network.graph.__class__()
        self.actor_graph.add_nodes_from(self.network.graph)
        self.actor_graph.add_edges_from(self.network.graph.edges())

    def make_geometry(self, render_throats=True, **kwargs):
        """Create VTK actors for all pores and throats in the network.

        One actor is created per pore (sphere) and per throat (cylinder).
        Actors are stored in ``self.actor_graph`` node/edge attributes.

        Parameters
        ----------
        render_throats : bool, optional
            If True, create throat actors. Default is True.
        **kwargs
            Additional keyword arguments:
            - ``pore_resolution`` : sphere mesh resolution (default 8).
            - ``throat_resolution`` : tube mesh resolution (default 6).
        """
        pore_resolution = kwargs.get("pore_resolution", 8)
        throat_resolution = kwargs.get("throat_resolution", 6)

        print(f"Creating actors (pore resolution={pore_resolution}, throat resolution={throat_resolution})")

        if self.network is not None:

            for n, props in self.network.graph.nodes.data():
                actor = create_pore_actor(resolution=pore_resolution, **props)
                self.actor_graph.nodes[n]["actor"] = actor

            if render_throats:
                for n1, n2, props in self.network.graph.edges.data():
                    c1 = np.array(self.network.graph.nodes[n1]["center"])
                    c2 = np.array(self.network.graph.nodes[n2]["center"])

                    if props.get("periodic", False):
                        ind = {"x": 0, "y": 1, "z": 2}
                        pdir = props.get("pdir", "x")
                        extent = self.network.graph.graph.get("extent")
                        if extent is not None:
                            if c1[ind[pdir]] > c2[ind[pdir]]:
                                c2[ind[pdir]] = extent[ind[pdir]] + c2[ind[pdir]]
                                c2 = 0.5 * (c1 + c2)
                            else:
                                c1[ind[pdir]] = extent[ind[pdir]] + c1[ind[pdir]]
                                c1 = 0.5 * (c1 + c2)

                    actor = create_throat_actor(c1, c2, resolution=throat_resolution, **props)
                    self.actor_graph[n1][n2]["actor"] = actor

        self.geom_complete = True

    def maps_colors(self, **kwargs):
        """Map colors and opacities to pore and throat actors.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments:
            - ``pores_colormap`` : dict mapping pore index to RGB color
              (default: {} -> all pores use DEFAULT_COLOR).
            - ``pores_opacitymap`` : dict mapping pore index to opacity
              (default: {} -> all pores use DEFAULT_OPACITY).
            - ``throats_colormap`` : list of RGB colors for throats, cycled
              if there are more throats than colors (default: [DEFAULT_COLOR]).
            - ``throats_opacitymap`` : list of opacity values for throats,
              cycled if needed (default: [DEFAULT_OPACITY]).
            - ``render_throats`` : if False, skip throat coloring
              (default: True).

        Notes
        -----
        Requires ``make_geometry`` to have been called first (``geom_complete``
        must be True).
        """
        from itertools import cycle

        if self.network is not None and self.geom_complete:
            if len(self.network.graph.nodes) != len(self.actor_graph.nodes):
                warn(
                    "The number of pore actors does not match the number of nodes in the pore network graph. "
                    "You should regenerate the geometry"
                )
                return

            print("Mapping colors")

            pores_colormap = kwargs.get("pores_colormap", {})
            pores_opacitymap = kwargs.get("pores_opacitymap", {})

            throats_colormap = kwargs.get("throats_colormap", [DEFAULT_COLOR])
            throats_opacitymap = kwargs.get("throats_opacitymap", [DEFAULT_OPACITY])

            if len(throats_colormap) == 0:
                throats_colormap = [DEFAULT_COLOR]
            if len(throats_opacitymap) == 0:
                throats_opacitymap = [DEFAULT_OPACITY]

            throats_colormap = cycle(throats_colormap)
            throats_opacitymap = cycle(throats_opacitymap)

            for pore, data in self.actor_graph.nodes.data():
                data["actor"].GetProperty().SetOpacity(pores_opacitymap.get(pore, DEFAULT_OPACITY))
                data["actor"].GetProperty().SetColor(pores_colormap.get(pore, DEFAULT_COLOR))

            if kwargs.get("render_throats", True):
                for n1, n2 in self.actor_graph.edges:
                    self.actor_graph[n1][n2]["actor"].GetProperty().SetColor(next(throats_colormap))
                    self.actor_graph[n1][n2]["actor"].GetProperty().SetOpacity(next(throats_opacitymap))

    def create_renWin(self, size=(800, 1000)):
        """Create a VTK render window and add the renderer.

        Parameters
        ----------
        size : tuple of int, optional
            Window size (width, height) in pixels. Default is (800, 1000).
        """
        if self.geom_complete:
            self.renWin = vtk.vtkRenderWindow()
            self.renWin.AddRenderer(self.renderer)
            self.renWin.SetSize(size)

    def close_renWin(self):
        """Close and delete the render window."""
        if self.renWin is not None:
            self.renWin.Finalize()
            del self.renWin
            self.renWin = None

    def render(self, azimuth=20, elevation=30, size=(800, 1000)):
        """Render the network interactively.

        Parameters
        ----------
        azimuth : float, optional
            Camera azimuth angle in degrees. Default is 20.
        elevation : float, optional
            Camera elevation angle in degrees. Default is 30.
        size : tuple of int, optional
            Window size (width, height). Default is (800, 1000).

        Notes
        -----
        Returns immediately if geometry has not been built (call ``make_geometry`` first).
        """
        if not self.geom_complete:
            return

        self.reset_renderer()
        self.create_renWin(size)

        for _, data in self.actor_graph.nodes.data():
            self.renderer.AddActor(data["actor"])

        for n1, n2 in self.actor_graph.edges:
            self.renderer.AddActor(self.actor_graph[n1][n2]["actor"])

        iren = vtk.vtkRenderWindowInteractor()
        self.set_camera(azimuth, elevation)
        self.renWin.OffScreenRenderingOff()
        iren.SetRenderWindow(self.renWin)
        iren.Initialize()

        iren.Start()
        del iren
        self.close_renWin()

    def save(self, filename, azimuth=20, elevation=30, size=(800, 1000)):
        """Render the network offscreen and save it as a JPEG image.

        Parameters
        ----------
        filename : str
            Output JPEG file path.
        azimuth : float, optional
            Camera azimuth angle. Default is 20.
        elevation : float, optional
            Camera elevation angle. Default is 30.
        size : tuple of int, optional
            Window size (width, height). Default is (800, 1000).

        Notes
        -----
        Returns immediately if geometry has not been built.
        """
        if not self.geom_complete:
            return

        self.create_renWin(size)
        self.set_camera(azimuth, elevation)
        self.renWin.OffScreenRenderingOn()
        self.renWin.Render()
        # Create the filter to write the rendered image
        w2i = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkJPEGWriter()
        writer.SetQuality(100)
        w2i.SetInput(self.renWin)
        w2i.Update()
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.SetFileName(filename)
        writer.Write()
        self.close_renWin()

    def reset_camera(self):
        """Restore the camera to its initial position (saved at reset_renderer time)."""
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(self.camera_props["pos"])
        cam.SetFocalPoint(self.camera_props["focal"])
        cam.SetViewUp(self.camera_props["viewup"])

    def set_camera(self, azimuth, elevation):
        """Position the camera with the given azimuth and elevation angles.

        Parameters
        ----------
        azimuth : float
            Camera azimuth angle in degrees.
        elevation : float
            Camera elevation angle in degrees.
        """
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().OrthogonalizeViewUp()
        self.renderer.GetActiveCamera().Azimuth(azimuth)
        self.renderer.GetActiveCamera().Elevation(elevation)
        self.renderer.ResetCameraClippingRange()


def create_sphere_glyphs(coord, radii, colors=None, resolution=8):
    """Create a single VTK actor rendering all pores as spheres using glyphs.

    Uses vtkGlyph3DMapper to instance a sphere source at each pore center,
    scaled by the pore radius. This is much faster than creating one actor
    per pore.

    Parameters
    ----------
    coord : list of array-like
        List of (x, y, z) pore center coordinates.
    radii : list of float
        List of pore radii. The sphere diameter is set to 2 * radius.
    colors : list of tuple or None, optional
        RGB colors for each pore (in node order). If None, no color array
        is added. Default is None.
    resolution : int, optional
        Sphere mesh resolution (theta and phi). Default is 8.

    Returns
    -------
    vtk.vtkActor
        A VTK actor containing all pore spheres as glyphs.
    """
    # Conversion des tableaux NumPy en objets VTK
    coord_vtk = vtk.vtkPoints()
    data = vtk.vtkPolyData()

    for i in range(len(coord)):
        coord_vtk.InsertNextPoint(coord[i])

    # Créez un tableau de données pour les rayons
    radiusData = vtk.vtkDoubleArray()
    radiusData.SetName("Radius")
    for r in radii:
        # scaling by the diameter
        radiusData.InsertNextValue(2 * r)

    # Associez les rayons aux points
    data.SetPoints(coord_vtk)
    data.GetPointData().AddArray(radiusData)

    # Créez un tableau de couleurs si elles sont fournies
    if colors is not None:
        colorsdata = vtk.vtkUnsignedCharArray()
        # colorsdata = vtk.vtkFloatArray()
        colorsdata.SetNumberOfComponents(3)
        colorsdata.SetName("Colors")

        for color in colors:
            colorsdata.InsertNextTuple(color)

        data.GetPointData().AddArray(colorsdata)

    # Créez une source de sphères pour les glyphes
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetThetaResolution(resolution)
    sphereSource.SetPhiResolution(resolution)

    # Créez un maillage de sphères pour les glyphes
    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

    # Créez un filtre vtkGlyph3DMapper
    glyphMapper = vtk.vtkGlyph3DMapper()
    glyphMapper.SetInputData(data)
    glyphMapper.SetSourceConnection(sphereSource.GetOutputPort())

    # Configurez les paramètres du glyphage
    glyphMapper.SetScaleArray("Radius")
    glyphMapper.SetScalarModeToUsePointFieldData()
    glyphMapper.SetScaleModeToScaleByMagnitude()
    glyphMapper.ScalarVisibilityOn()
    glyphMapper.SelectColorArray("Colors")

    # Créez un acteur pour les glyphes
    glyphActor = vtk.vtkActor()
    glyphActor.SetMapper(glyphMapper)

    return glyphActor


def create_tubes(heads, tails, radii, colors=None, resolution=8):
    """Create a single VTK actor rendering all throats as tubes.

    Uses vtkTubeFilter to instance cylinders along line segments between
    throat endpoints, with radius varying per point. This is much faster
    than creating one actor per throat.

    Parameters
    ----------
    heads : list of array-like
        List of (x, y, z) coordinates for the head (first endpoint) of each
        tube segment.
    tails : list of array-like
        List of (x, y, z) coordinates for the tail (second endpoint) of each
        tube segment.
    radii : list of float
        List of tube radii (one value per endpoint, i.e. 2 per throat).
    colors : list of tuple or None, optional
        RGB colors for each tube endpoint. If None, no color array is added.
        Default is None.
    resolution : int, optional
        Tube mesh resolution (number of sides). Default is 8.

    Returns
    -------
    vtk.vtkActor
        A VTK actor containing all throat tubes.
    """
    # Conversion des tableaux NumPy en objets VTK

    polydata = vtk.vtkPolyData()
    points = list(zip(tails, heads))
    points = np.vstack(points)
    pairs = np.arange(len(tails) * 2).reshape(-1, 2)
    # radii = np.repeat(radii, 2)

    # assert (points.size // 3 == pairs.size)
    # assert (pairs.size == radii.size)

    pointArray = vtk.vtkPoints()
    for (x, y, z) in points:
        pointArray.InsertNextPoint(x, y, z)
    polydata.SetPoints(pointArray)

    cellArray = vtk.vtkCellArray()
    for ids in pairs:
        idList = vtk.vtkIdList()
        for i in ids:
            idList.InsertNextId(i)
        cellArray.InsertNextCell(idList)
    polydata.SetLines(cellArray)

    floats = vtk.vtkFloatArray()
    for r in radii:
        floats.InsertNextValue(r)
    polydata.GetPointData().SetScalars(floats)

    if colors is not None:
        colorsdata = vtk.vtkUnsignedCharArray()
        # colorsdata = vtk.vtkFloatArray()
        colorsdata.SetNumberOfComponents(3)
        colorsdata.SetName("Colors")
        for color in colors:
            # One tuple per endpoint point: ``colors`` already carries two
            # entries (head, tail) per throat segment. Inserting each entry
            # twice made the array 2x too long, so VTK read it by point index
            # and every segment after the first inherited a neighbor's color.
            colorsdata.InsertNextTuple(color)

        polydata.GetPointData().AddArray(colorsdata)

    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(polydata)
    tubeFilter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tubeFilter.SetNumberOfSides(resolution)
    # self.tubeFilter.CappingOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubeFilter.GetOutputPort())
    if colors is not None:
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray("Colors")
    mapper.ScalarVisibilityOn()
    actor = vtk.vtkActor()

    actor.SetMapper(mapper)

    return actor


def create_pore_actor(center, radius, resolution=8, **kwargs):
    """Create a single VTK actor for one pore (sphere).

    Parameters
    ----------
    center : array-like of float
        (x, y, z) coordinates of the pore center.
    radius : float
        Pore radius.
    resolution : int, optional
        Sphere mesh resolution (theta and phi). Default is 8.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    vtk.vtkActor
        A VTK actor containing a single sphere at the pore center.
    """
    points = vtk.vtkPoints()
    points.InsertNextPoint(center[0], center[1], center[2])
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    sphere = vtk.vtkSphereSource()
    sphere.SetThetaResolution(resolution)
    sphere.SetPhiResolution(resolution)
    sphere.SetRadius(radius)
    # sphere.SetCenter(list(center))

    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetSourceConnection(sphere.GetOutputPort())
    glyph3D.SetInputData(polydata)
    glyph3D.Update()

    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(glyph3D.GetOutputPort())
    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    return sphereActor


def create_throat_actor(c1, c2, radius, resolution=6, **kwargs):
    """Create a single VTK actor for one throat (cylinder).

    Parameters
    ----------
    c1 : array-like of float
        (x, y, z) coordinates of the first pore center.
    c2 : array-like of float
        (x, y, z) coordinates of the second pore center.
    radius : float
        Throat radius.
    resolution : int, optional
        Tube mesh resolution (number of sides). Default is 6.
    **kwargs
        Additional keyword arguments (currently unused).

    Returns
    -------
    vtk.vtkActor
        A VTK actor containing a single cylinder between c1 and c2.
    """
    line = vtk.vtkLineSource()
    line.SetPoint1(c1[0], c1[1], c1[2])
    line.SetPoint2(c2[0], c2[1], c2[2])
    cylinder = vtk.vtkTubeFilter()
    cylinder.SetInputConnection(line.GetOutputPort())
    cylinder.SetRadius(radius)
    cylinder.SetNumberOfSides(resolution)  # nombre de cotés du tube
    cylinderMapper = vtk.vtkPolyDataMapper()
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())
    cylinderActor = vtk.vtkActor()
    cylinderActor.SetMapper(cylinderMapper)
    return cylinderActor


def vtk_viewer(
    pn,
    save=True,
    render=True,
    camera=(20, 30),
    cmap="viridis",
    filename=None,
    resolution=8,
    attr="radius",
    high=None,
    low=None,
    throat_colors=None,
    periodic=True,
):
    """Generate a 3D rendering of the given pore network using Fast3DViewer.

    Parameters
    ----------
    pn : pore_network
        The pore network model instance to visualize.
    save : bool, optional
        If True, save the visualization to a JPEG file. Default is True. When
        ``render`` is also True, the image uses the camera state adjusted
        during interaction (see :meth:`Fast3DViewer.save_current_view`);
        otherwise it uses the ``camera`` angles.
    render : bool, optional
        If True, render the network in an interactive VTK window. If False,
        the network is rendered offscreen only. Default is True.
    camera : tuple of float, optional
        (azimuth, elevation) camera angles in degrees. Default is (20, 30).
    cmap : str, optional
        Matplotlib colormap name used to map a given attribute to color.
        Default is "viridis".
    filename : str or None, optional
        Output JPEG file path. If None, a default name is generated from the
        network size and extent. Default is None.
    resolution : int, optional
        Mesh resolution for pores and throats. Default is 8.
    attr : str, optional
        Node/edge attribute used for coloring (default: "radius").
    high : float or None, optional
        Maximum value for colormap normalization. If None, uses the max of
        the attribute values. Default is None.
    low : float or None, optional
        Minimum value for colormap normalization. If None, uses the min of
        the attribute values. Default is None.
    throat_colors : None, array-like of length n_edges, or other value, optional
        Throat coloring mode (default is None):

        - ``None``: use the edge values of ``attr`` if present; otherwise each
          throat is colored with the mean of its endpoint nodes so node-only
          fields (e.g. a solved concentration) still color the throats.
        - Array-like with one scalar per throat (in edge order): those values
          are used directly, mapped through the same colormap/normalization.
        - Any other value (e.g. True): force the pore-mean coloring even if
          ``attr`` exists on edges.
    periodic : bool, optional
        Coloring mode for periodic throats (default is True). If True, both
        halves of a periodic throat share its per-edge color — for node-only
        fields that is the mean of the two endpoint pores. If False, each
        half-capillary crossing a box face is colored with the color of the real
        pore it connects to on that side of the box (so a stub exiting the +x
        face takes the color of its +x-face pore). Default is True.

    Notes
    -----
    Requires node/edge attribute ``attr`` to exist on the graph (e.g. ``radius``).

    Raises
    ------
    ValueError
        If ``throat_colors`` is an array-like whose length differs from the
        number of throats.
    """
    pattrval = list(nx.get_node_attributes(pn.graph, attr).values())
    tattrval = list(nx.get_edge_attributes(pn.graph, attr).values())

    def _pore_mean():
        """Per-throat value as the mean of its endpoint nodes (edge order)."""
        return [0.5 * (pn.graph.nodes[u][attr] + pn.graph.nodes[v][attr]) for u, v in pn.graph.edges()]

    if throat_colors is None:
        # Node-only attribute (e.g. a solved concentration field): interpolate each
        # throat value as the mean of its endpoint nodes so throats follow the field.
        if len(tattrval) == 0 and len(pattrval) > 0:
            tattrval = _pore_mean()
    else:
        try:
            explicit = [float(v) for v in throat_colors]
        except (TypeError, ValueError):
            explicit = None
        if explicit is not None:
            n_edges = pn.graph.number_of_edges()
            if len(explicit) != n_edges:
                raise ValueError(
                    f"throat_colors has {len(explicit)} values; expected {n_edges} "
                    "(one scalar per throat, in edge order)."
                )
            tattrval = explicit
        elif len(pattrval) > 0:
            # Non-iterable flag (e.g. True): force pore-mean coloring even if attr exists on edges.
            tattrval = _pore_mean()

    all_vals = pattrval + tattrval
    if len(all_vals) == 0:
        warn(f"No '{attr}' attribute found on nodes or edges. Cannot color the network.")
        return

    vmin = low if low is not None else min(all_vals)
    vmax = high if high is not None else max(all_vals)
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap)

    pores_cm = [
        np.around(np.array(cmap_obj(norm(r))[:-1]) * 255).astype(np.uint8)
        for r in pattrval
    ]
    throats_cm = [
        np.around(np.array(cmap_obj(norm(r))[:-1]) * 255).astype(np.uint8)
        for r in tattrval
    ]

    viewer = Fast3DViewer(pn)

    if render:
        viewer.render(pcmap=pores_cm, tcmap=throats_cm, resolution=resolution, periodic=periodic)
        viewer.interact(azimuth=camera[0], elevation=camera[1])

    if save:
        if filename is None:
            extent = pn.graph.graph.get("extent")
            if extent is not None:
                filename = "PNM-{0:d}n-{1:d}e-{2:.2f}x{3:.2f}x{4:.2f}.jpg".format(
                    pn.graph.number_of_nodes(), pn.graph.number_of_edges(), extent[0], extent[1], extent[2]
                )
            else:
                filename = "PNM-{0:d}n-{1:d}e.jpg".format(
                    pn.graph.number_of_nodes(), pn.graph.number_of_edges()
                )
        if render:
            # Preserve the camera state adjusted during interaction.
            viewer.save_current_view(filename)
        else:
            viewer.save(
                filename,
                pcmap=pores_cm,
                tcmap=throats_cm,
                resolution=resolution,
                azimuth=camera[0],
                elevation=camera[1],
                periodic=periodic,
            )

