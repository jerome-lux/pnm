# -*- coding: utf-8

import vtk
from warnings import warn
import numpy as np
import networkx as nx
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import pnm


# TODO : change throats_opacitymap and throats_colormap to something else than list...

DEFAULT_COLOR = (1, 1, 1)
DEFAULT_OPACITY = 1

class Fast3DViewer:
    def __init__(self, network):
        self.network = network
        self.pore_actor = None
        self.throat_actor = None

    @classmethod
    def fromfile(cls, filename):
        return cls(network=pnm.pore_network.read(filename))

    def reset_renderer(self, bgcolor=(1, 1, 1)):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(bgcolor)
        cam = self.renderer.GetActiveCamera()
        self.camera_props = dict(
            zip(("pos", "focal", "viewup"), (cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp()))
        )

    def create_pores(self, colors=None, resolution=8, **kwargs):

        print(f"Creating pore glyph actor (resolution={resolution})")

        centers = list(nx.get_node_attributes(self.network.graph, "center").values())
        radii = list(nx.get_node_attributes(self.network.graph, "radius").values())

        self.pore_actor = create_sphere_glyphs(coord=centers, radii=radii, colors=colors)


    def create_throats(self, colors=None, resolution=6):

        heads = []
        tails = []
        radii = []

        for n1, n2, props in self.network.graph.edges.data():
            c1 = np.array(self.network.graph.nodes[n1]["center"])
            c2 = np.array(self.network.graph.nodes[n2]["center"])
            radii.append(props["radius"])

            if props["periodic"]:
                ind = {"x": 0, "y": 1, "z": 2}
                pdir = props["pdir"]
                if c1[ind[pdir]] > c2[ind[pdir]]:
                    c2[ind[pdir]] = self.network.graph.graph["extent"][ind[pdir]] + c2[ind[pdir]]
                    c2 = 0.5 * (c1 + c2)
                else:
                    c1[ind[pdir]] = self.network.graph.graph["extent"][ind[pdir]] + c1[ind[pdir]]
                    c1 = 0.5 * (c1 + c2)
            heads.append(c1)
            tails.append(c2)

        self.throat_actor = create_tubes(heads, tails, radii, colors, resolution=resolution)


    def create_renWin(self, size=(800, 1000)):
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.renderer)
        self.renWin.SetSize(size)

    def close_renWin(self):
        self.renWin.Finalize()
        del self.renWin
        self.renWin = None


    def render(self, pcmap=None, tcmap=None, resolution=8, azimuth=20, elevation=30, size=(800, 1000)):

        self.reset_renderer()
        self.create_renWin(size)

        if self.pore_actor is None:
            self.create_pores(colors=pcmap, resolution=resolution)
        self.renderer.AddActor(self.pore_actor)

        if self.throat_actor is None:
            self.create_throats(colors=tcmap, resolution=resolution)
        self.renderer.AddActor(self.throat_actor)

        iren = vtk.vtkRenderWindowInteractor()
        self.set_camera(azimuth, elevation)
        self.renWin.OffScreenRenderingOff()
        iren.SetRenderWindow(self.renWin)
        iren.Initialize()

        iren.Start()
        del iren
        self.close_renWin()

    def reset_camera(self):
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(self.camera_props["pos"])
        cam.SetFocalPoint(self.camera_props["focal"]),
        cam.SetViewUp(self.camera_props["viewup"])

    def set_camera(self, azimuth, elevation):
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().OrthogonalizeViewUp()
        self.renderer.GetActiveCamera().Azimuth(azimuth)
        self.renderer.GetActiveCamera().Elevation(elevation)
        self.renderer.ResetCameraClippingRange()

    def save(self, filename, pcmap=None, tcmap=None, resolution=8, azimuth=20, elevation=30, size=(800, 1000)):

        self.create_renWin(size)
        # self.reset_camera()
        self.set_camera(azimuth, elevation)

        if self.pore_actor is None:
            self.create_pores(colors=pcmap, resolution=resolution)
        self.renderer.AddActor(self.pore_actor)

        if self.throat_actor is None:
            self.create_throats(colors=tcmap, resolution=resolution)
        self.renderer.AddActor(self.throat_actor)

        # self.renWin.AddRenderer(self.renderer)
        self.renWin.OffScreenRenderingOn()
        self.renWin.Render()
        # Création du filtre pour écriture de l'image du rendu
        w2i = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkJPEGWriter()
        writer.SetQuality(100)
        w2i.SetInput(self.renWin)
        w2i.Update()
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.SetFileName(filename)
        writer.Write()
        self.close_renWin()


class PNM_3d_viewer:
    def __init__(self, network=None):
        self.set_network(network)

        self.geom_complete = False

    @classmethod
    def fromfile(cls, filename):
        return cls(network=pnm.pore_network.read(filename))

    def reset_renderer(self, bgcolor=(1, 1, 1)):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(bgcolor)
        cam = self.renderer.GetActiveCamera()
        self.camera_props = dict(
            zip(("pos", "focal", "viewup"), (cam.GetPosition(), cam.GetFocalPoint(), cam.GetViewUp()))
        )

    def set_network(self, network):
        self.network = network
        self.actor_graph = self.network.graph.__class__()
        self.actor_graph.add_nodes_from(self.network.graph)
        # edges = [(e1, e2) for (e1, e2) in self.network.graph.edges() if not self.network.graph[e1][e2]["periodic"]]
        self.actor_graph.add_edges_from(self.network.graph.edges())
        # self.actor_graph.add_edges_from(edges)

    def make_geometry(self, render_throats=True,**kwargs):
        pore_resolution = kwargs.get("pore_resolution", 8)
        throat_resolution = kwargs.get("throat_resolution", 6)

        print(f"Creating actors (pore resolution={pore_resolution}, throat resolution={throat_resolution})")

        if self.network is not None:

            for n, props in self.network.graph.nodes.data():
                actor = create_pore_actor(resolution=pore_resolution, **props)
                self.actor_graph.nodes[n]["actor"] = actor
                # self.renderer.AddActor(actor)

            if render_throats:
                for n1, n2, props in self.network.graph.edges.data():
                    c1 = np.array(self.network.graph.nodes[n1]["center"])
                    c2 = np.array(self.network.graph.nodes[n2]["center"])

                    if props["periodic"]:
                        ind = {"x": 0, "y": 1, "z": 2}
                        pdir = props["pdir"]
                        if c1[ind[pdir]] > c2[ind[pdir]]:
                            c2[ind[pdir]] = self.network.graph.graph["extent"][ind[pdir]] + c2[ind[pdir]]
                            c2 = 0.5 * (c1 + c2)
                        else:
                            c1[ind[pdir]] = self.network.graph.graph["extent"][ind[pdir]] + c1[ind[pdir]]
                            c1 = 0.5 * (c1 + c2)

                    actor = create_throat_actor(c1, c2, resolution=throat_resolution, **props)
                    self.actor_graph[n1][n2]["actor"] = actor
                    # self.renderer.AddActor(actor)

        self.geom_complete = True

    def maps_colors(self, **kwargs):
        """pores_colormap : dict of color to be mapped to the pores, if only one value is provided, this color is applied to all elements
        pores_opacitymap : dict of opacity values for pores
        throats_colormap : list of color values for throats
        throats_opacitymap : list of opacity values for throats
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
                # if len(self.network.graph.edges) != len(self.actor_graph.edges):
                #     warn("The number of throat actors does not match the number of edges in the pore network graph. "
                #          "You should regenerate the geometry")
                #     return

                for n1, n2 in self.actor_graph.edges:
                    self.actor_graph[n1][n2]["actor"].GetProperty().SetColor(next(throats_colormap))
                    self.actor_graph[n1][n2]["actor"].GetProperty().SetOpacity(next(throats_opacitymap))

    def create_renWin(self, size=(800, 1000)):
        if self.geom_complete:
            self.renWin = vtk.vtkRenderWindow()
            self.renWin.AddRenderer(self.renderer)
            self.renWin.SetSize(size)

    def close_renWin(self):
        self.renWin.Finalize()
        del self.renWin
        self.renWin = None

    def render(self, azimuth=20, elevation=30, size=(800, 1000)):

        if not self.geom_complete:
            return
        # self.reset_camera()

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
        if not self.geom_complete:
            return

        self.create_renWin(size)
        # self.reset_camera()
        self.set_camera(azimuth, elevation)
        # self.renWin.AddRenderer(self.renderer)
        self.renWin.OffScreenRenderingOn()
        self.renWin.Render()
        # Création du filtre pour écriture de l'image du rendu
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
        cam = self.renderer.GetActiveCamera()
        cam.SetPosition(self.camera_props["pos"])
        cam.SetFocalPoint(self.camera_props["focal"]),
        cam.SetViewUp(self.camera_props["viewup"])

    def set_camera(self, azimuth, elevation):
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().OrthogonalizeViewUp()
        self.renderer.GetActiveCamera().Azimuth(azimuth)
        self.renderer.GetActiveCamera().Elevation(elevation)
        self.renderer.ResetCameraClippingRange()


def create_sphere_glyphs(coord, radii, colors=None, resolution=8):
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
        radiusData.InsertNextValue(2*r)

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
    # Conversion des tableaux NumPy en objets VTK

    polydata = vtk.vtkPolyData()
    points = np.vstack(zip(tails, heads))
    pairs = np.arange(len(tails)*2).reshape(-1, 2)
    radii = np.repeat(radii, 2)

    assert (points.size/3. == pairs.size)
    assert (pairs.size == radii.size)

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
            # Début et fin du tube
            colorsdata.InsertNextTuple(color)
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
    pn,  save=True, render=True, cmap="viridis", filename=None, resolution=8, attr="radius", high=None, low=None
):
    """
    Generate a 3D rendering of the given pore network.
    pn: pore network instance
    render: if True a the network is rendered in an interactive vtk window. It is rendered offscreen otherwise
    save: the visualizaton is saved
    cmap: colormap to map a given attribute (default radius)
    filename: save file name
    attr: each element is colored using the value of the given attribute (default: radius)
    high: max value for normalization
    low: min value for normalization
    """

    pattrval = nx.get_node_attributes(pn.graph, attr).values()
    tattrval = nx.get_edge_attributes(pn.graph, attr).values()

    norm = Normalize(vmin=min(min(pattrval), min(tattrval)), vmax=max(max(pattrval), max(tattrval)))

    cmap = plt.get_cmap(cmap)

    pores_cm = [np.around(np.array(cmap(norm(r))[:-1])*255).astype(np.uint8) for r in nx.get_node_attributes(pn.graph, attr).values()]
    throats_cm = [np.around(np.array(cmap(norm(r))[:-1])*255).astype(np.uint8) for r in nx.get_edge_attributes(pn.graph, attr).values()]

    viewer = Fast3DViewer(pn)

    if render:
        viewer.render(pcmap=pores_cm, tcmap=throats_cm, resolution=resolution)

    if save:
        if filename is None:
            extent = pn.graph.graph["extent"]
            filename = "PNM-{0:d}n-{1:d}e-{2:.2f}x{3:.2f}x{4:.2f}.jpg".format(
                pn.graph.number_of_nodes(), pn.graph.number_of_edges(), extent[0], extent[1], extent[2]
            )
        viewer.save(filename, pcmap=pores_cm, tcmap=throats_cm, resolution=resolution)

