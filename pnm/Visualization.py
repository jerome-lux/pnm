# -*- coding: utf-8

import vtk
from warnings import warn
import numpy as np
import networkx as nx

#TODO : change throats_opacitymap and throats_colormap to something else than list...

DEFAULT_COLOR = (1,1,1)
DEFAULT_OPACITY = 1

class PNM_3d_viewer():

    def __init__(self,network=None):

        self.set_network(network)

        self.geom_complete = False

    @classmethod
    def fromfile(cls,filename):

        return cls(network = pnm.pore_network.read(filename))

    def reset_renderer(self,bgcolor=(1,1,1)):

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(bgcolor)
        cam = self.renderer.GetActiveCamera()
        self.camera_props = dict(zip(('pos','focal','viewup'),
                                (cam.GetPosition(),cam.GetFocalPoint(),cam.GetViewUp()) ))

    def set_network(self,network):

        self.network = network
        self.actor_graph = self.network.graph.__class__()
        self.actor_graph.add_nodes_from(self.network.graph)
        self.actor_graph.add_edges_from(self.network.graph.edges())

    def make_geometry(self,**kwargs):

        pore_resolution = kwargs.get('pore_resolution',8)
        throat_resolution = kwargs.get('throat_resolution',6)

        if self.network is not None:

            self.reset_renderer()

            for n,props in self.network.graph.nodes.data():
                actor = create_pore_actor(resolution=pore_resolution, **props)
                self.actor_graph.nodes[n]['actor'] = actor
                self.renderer.AddActor(actor)

            if kwargs.get('render_throats',True):
                for n1,n2,props in self.network.graph.edges.data():
                    c1 = self.network.graph.nodes[n1]['center']
                    c2 = self.network.graph.nodes[n2]['center']
                    actor = create_throat_actor(c1,c2,resolution=throat_resolution, **props)
                    self.actor_graph[n1][n2]['actor'] = actor
                    self.renderer.AddActor(actor)

        self.geom_complete = True

    def maps_colors(self,**kwargs):

        """pores_colormap : dict of color to be mapped to the pores, if only one value is provided, this color is applied to all elements
            pores_opacitymap : dict of opacity values for pores
            throats_colormap : list of color values for throats
            throats_opacitymap : list of opacity values for throats
        """
        from itertools import cycle

        if self.network is not None and self.geom_complete :

            if len(self.network.graph.nodes) != len(self.actor_graph.nodes):
                warn("The number of pore actors does not match the number of nodes in the pore network graph. "
                    "You should regenerate the geometry")
                return

            pores_colormap = kwargs.get('pores_colormap',{})
            pores_opacitymap = kwargs.get('pores_opacitymap',{})


            throats_colormap = kwargs.get('throats_colormap',[DEFAULT_COLOR])
            throats_opacitymap = kwargs.get('throats_opacitymap',[DEFAULT_OPACITY])

            if len(throats_colormap) == 0: throats_colormap = [DEFAULT_COLOR]
            if len(throats_opacitymap) == 0: throats_opacitymap = [DEFAULT_OPACITY]

            throats_colormap = cycle(throats_colormap)
            throats_opacitymap = cycle(throats_opacitymap)

            for pore,data in self.actor_graph.nodes.data():
                data['actor'].GetProperty().SetOpacity(pores_opacitymap.get(pore,DEFAULT_OPACITY))
                data['actor'].GetProperty().SetColor(pores_colormap.get(pore,DEFAULT_COLOR))

            if kwargs.get('render_throats',True):
                if len(self.network.graph.edges) != len(self.actor_graph.edges):
                    warn("The number of throat actors does not match the number of edges in the pore network graph. "
                        "You should regenerate the geometry")
                    return

                for n1,n2 in self.actor_graph.edges:

                    self.actor_graph[n1][n2]['actor'].GetProperty().SetColor(next(throats_colormap))
                    self.actor_graph[n1][n2]['actor'].GetProperty().SetOpacity(next(throats_opacitymap))

    def create_renWin(self,size=(800,1000)):

        if self.geom_complete:
            self.renWin = vtk.vtkRenderWindow()
            self.renWin.AddRenderer(self.renderer)
            self.renWin.SetSize(size)

    def close_renWin(self):

        self.renWin.Finalize()
        del (self.renWin)
        self.renWin = None

    def render(self,azimuth=20,elevation=30,size=(800,1000)):

        if not self.geom_complete:
            return
        #self.reset_camera()
        self.create_renWin(size)
        iren = vtk.vtkRenderWindowInteractor()
        self.set_camera(azimuth,elevation)
        self.renWin.OffScreenRenderingOff()
        iren.SetRenderWindow(self.renWin)
        iren.Initialize()
        iren.Start()
        del (iren)
        self.close_renWin()

    def save(self,filename,azimuth=20,elevation=30,size=(800,1000)):

        if not self.geom_complete:
            return

        self.create_renWin(size)
        #self.reset_camera()
        self.set_camera(azimuth,elevation)
        # self.renWin.AddRenderer(self.renderer)
        self.renWin.OffScreenRenderingOn()
        self.renWin.Render()
        w2i = vtk.vtkWindowToImageFilter()   #Création du filtre pour écriture de l'image du rendu
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
        cam.SetPosition(self.camera_props['pos'])
        cam.SetFocalPoint(self.camera_props['focal']),
        cam.SetViewUp(self.camera_props['viewup'])

    def set_camera(self,azimuth,elevation):

        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().OrthogonalizeViewUp()
        self.renderer.GetActiveCamera().Azimuth(azimuth)
        self.renderer.GetActiveCamera().Elevation(elevation)
        self.renderer.ResetCameraClippingRange()


#def create_pore_actor_old(center, radius, resolution = 8, **kwargs):

    #sphere = vtk.vtkSphereSource()
    #sphere.SetThetaResolution(resolution)
    #sphere.SetPhiResolution(resolution)
    #sphere.SetRadius(radius)
    #sphere.SetCenter(list(center))
    #sphereMapper = vtk.vtkPolyDataMapper()
    #sphereMapper.SetInputConnection(sphere.GetOutputPort())
    #sphereActor = vtk.vtkActor()
    #sphereActor.SetMapper(sphereMapper)
    #return(sphereActor)

def create_pore_actor(center, radius, resolution = 8, **kwargs):

    points = vtk.vtkPoints()
    points.InsertNextPoint(center[0],center[1],center[2])
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    # Create anything you want here, we will use a cube for the demo.

    sphere = vtk.vtkSphereSource()
    #sphere.SetThetaResolution(resolution)
    #sphere.SetPhiResolution(resolution)
    sphere.SetRadius(radius)
    #sphere.SetCenter(list(center))

    glyph3D = vtk.vtkGlyph3D()
    glyph3D.SetSourceConnection(sphere.GetOutputPort())
    glyph3D.SetInputData(polydata)
    glyph3D.Update()


    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(glyph3D.GetOutputPort())
    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    return(sphereActor)

def create_throat_actor(c1, c2, radius, resolution = 6, **kwargs):

    line = vtk.vtkLineSource()
    line.SetPoint1(c1[0],c1[1],c1[2])
    line.SetPoint2(c2[0],c2[1],c2[2])
    cylinder = vtk.vtkTubeFilter()
    cylinder.SetInputConnection(line.GetOutputPort())
    cylinder.SetRadius(radius)
    cylinder.SetNumberOfSides(resolution)                        #nombre de cotés du tube
    cylinderMapper = vtk.vtkPolyDataMapper()
    cylinderMapper.SetInputConnection(cylinder.GetOutputPort())
    cylinderActor = vtk.vtkActor()
    cylinderActor.SetMapper(cylinderMapper)
    return (cylinderActor)


def vtk_viewer(pn,render=True,save=True,cmap = 'viridis',filename=None,render_throats=True,attr='radius',high=None,low=None):

    """
    Generate a 3D rendering of the given pore network.
    pn: pore network instance
    render: if True a the network is rendered in an interactive vtk window. It is rendered offscreen otherwise
    save: the visualizaton is saved
    cmap: colormap to map a given attribute (default radius)
    filename: save file name
    render_throats: if True throats are drawn
    attr: each element is colored using the value of the given attribute (default: radius)
    high: max value for normalization
    low: min value for normalization
    """

    import networkx as nx
    import numpy as np
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    if high is None or low is None:
        try:
            attrval = nx.get_node_attributes(pn.graph,attr).values()
        except:
            warn("Pore attribute {} does not exist. Exit".format(attr))
            return

    if high is None:
        maxval = max(attrval)
    else:
        maxval = high

    if low is None:
        minval = min(attrval)
    else:
        minval = low

    norm = Normalize(vmin=minval, vmax=maxval)

    cmap = plt.get_cmap(cmap)

    pores_cm = {n:cmap(norm(r))[:-1] for (n,r) in nx.get_node_attributes(pn.graph,attr).items()}
    throats_cm = [cmap(norm(r))[:-1] for r in nx.get_edge_attributes(pn.graph,attr).values()]

    viewer = PNM_3d_viewer(pn)
    viewer.make_geometry(pore_resolution = 8, throat_resolution = 4,render_throats=render_throats)
    viewer.maps_colors(pores_colormap = pores_cm,throats_colormap=throats_cm,render_throats=render_throats)

    if render:
        viewer.render()
    if save:
        if filename is None:
            extent = pn.graph.graph['extent']
            filename = "PNM-{0:d}n-{1:d}e-{2:.2f}x{3:.2f}x{4:.2f}.jpg".format(pn.graph.number_of_nodes(),pn.graph.number_of_edges(),extent[0],extent[1],extent[2])
        viewer.save(filename)


def Fast3DViewer(net,opacity=0.2, cmap = 'viridis',attr = 'radius',throats_radius = False, vmax = None, vmin = None):

    """Quick way to vizualize big network, throats are rendered as lines
    Pores are colored by their radii"""

    from mayavi import mlab
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    #Nodes
    x,y,z = np.array(list(nx.get_node_attributes(net.graph,'center').values())).T

    attrval = np.array(list(nx.get_node_attributes(net.graph,attr).values()))

    pores = mlab.points3d(x, y, z, 2*attrval, colormap=cmap, scale_factor=1,scale_mode='scalar',vmin=vmin,vmax=vmax)

    pores.mlab_source.dataset.lines = np.array(net.graph.edges())
    if throats_radius:
        tube = mlab.pipeline.tube(pores, tube_radius = throats_radius)
        tube.filter.radius_factor = 0.1
        tube.filter.vary_radius = 'vary_radius_by_absolute_scalar'
        mlab.pipeline.surface(tube, colormap="viridis", opacity=opacity, line_width=1)
    else:
        mlab.pipeline.surface(pores, colormap="viridis", opacity=opacity, line_width=1)
    mlab.show()




"""PLOTLY jupyter
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import numpy as np

x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 400).transpose()


trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=np.abs(z)*10,
        color=z,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='3d-scatter-colorscale')
"""



"""#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkSphereSource.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkActor.h>
#include <vtkPoints.h>
#include <math.h>
#include <vtkPointData.h>
#include <vtkFloatArray.h>
#include <vtkGlyph3D.h>
#include <vtkUnstructuredGrid.h>
#include <vtkLookupTable.h>

#define PI 3.14159265

int main(int, char *[])
{
    srand(time(NULL));

    // create points
    vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New();

    // setup scales
    vtkSmartPointer<vtkFloatArray> scales = vtkSmartPointer<vtkFloatArray>::New();
    scales->SetName("scales");

     // setup color label
    vtkSmartPointer<vtkFloatArray> col = vtkSmartPointer<vtkFloatArray>::New();
    col->SetName("col");

    // setup lookupTable and add some colors
    vtkSmartPointer<vtkLookupTable> colors = vtkSmartPointer<vtkLookupTable>::New();
    colors->SetNumberOfTableValues(4);
    colors->SetTableValue(0 ,1.0 ,0.0 ,0.0 ,1.0); // red
    colors->SetTableValue(1 ,0.0 ,1.0 ,0.0 ,1.0); // green
    colors->SetTableValue(2 ,0.0 ,0.0 ,1.0 ,1.0); // blue
    colors->SetTableValue(3 ,1.0 ,1.0 ,0.0 ,1.0); // yellow
    // the last double value is for opacity (1->max, 0->min)

    for(int i=0; i<100; i++)
    {
        points->InsertNextPoint(15*cos(i*PI/50), 15*sin(i*PI/50), 0); // sphere in circle
        scales->InsertNextValue((rand()% 100)/double(100)); // random radius between 0 and 0.99
        col->InsertNextValue((rand()% 4)); // random color label
    }

    // grid structured to append center, radius and color label
    vtkSmartPointer<vtkUnstructuredGrid> grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
    grid->SetPoints(points);
    grid->GetPointData()->AddArray(scales);
    grid->GetPointData()->SetActiveScalars("scales"); // !!!to set radius first
    grid->GetPointData()->AddArray(col);

    // create anything you want here, we will use a sphere for the demo
    vtkSmartPointer<vtkSphereSource> sphereSource = vtkSmartPointer<vtkSphereSource>::New();

    // object to group sphere and grid and keep smooth interaction
    vtkSmartPointer<vtkGlyph3D> glyph3D = vtkSmartPointer<vtkGlyph3D>::New();
    glyph3D->SetInputData(grid);
    glyph3D->SetSourceConnection(sphereSource->GetOutputPort());

    // create a mapper and actor
    vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    mapper->SetInputConnection(glyph3D->GetOutputPort());

    mapper->SetScalarModeToUsePointFieldData(); // without, color are displayed regarding radius and not color label
    mapper->SetScalarRange(0, 3); // to scale color label (without, col should be between 0 and 1)
    mapper->SelectColorArray("col"); // !!!to set color (nevertheless you will have nothing)
    mapper->SetLookupTable(colors);

    vtkActor *actor = vtkActor::New();
    actor->SetMapper(mapper);

    // create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);
    // add the actor to the scene
    renderer->AddActor(actor);
    renderer->SetBackground(0, 0, 0);
    // render and interact
    renderWindow->Render();
    renderWindowInteractor->Start();
    return EXIT_SUCCESS;
}
"""
