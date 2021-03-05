#!/usr/bin/python
# -*- coding: utf-8

'''

2018 : works with python 3x vtk8

'''
#
import pnm

import time
import string
import re
import sys
import os
import glob
from optparse import OptionParser

parser = OptionParser()

parser.add_option("-f", "--file", dest="filename",default=None,
                                    help="FILE to process. If not present all files with .network extension are processed", metavar="FILE")
parser.add_option("--attr", dest="attr",default="radius", help="attribute to map color from")
parser.add_option("-c","--camera",dest="camera",nargs=2,type="float", help="Position of camera: elevation rotation",metavar="E R")
parser.add_option("-o","--opacity",action="store_true",dest="opacity",help="Make pore with max attr value partially transparent")
parser.add_option("-s","--size",dest="size",nargs=2,help="Size of the render window (lx ly)", metavar="lx ly", type="int")
parser.add_option("-i","--interact",dest="interact",action="store_true", help="Enable interacting mode")
parser.add_option("-p","--point",type="int",dest="point",action="append", help="highlight specified pores")
parser.add_option("--cmap",dest="cmap",default="Blues",help="Colormap name (see matplotlib colormap)")
parser.add_option("--scale",type="float",dest="scale",default=1,help="scale factor")
parser.add_option("--vmin",type="float",dest="vmin",default=None,help="min value for color normalization")
parser.add_option("--vmax",type="float",dest="vmax",default=None,help="max value for color normalization")
parser.add_option("--xperiodic", dest="xperiodic", action="store_true", help="Network is periodic in the x direction")
parser.add_option("--yperiodic", dest="yperiodic", action="store_true", help="Network is periodic in the y direction")
parser.add_option("--zperiodic", dest="zperiodic", action="store_true", help="Network is periodic in the z direction")

parser.add_option("--box", dest="box", nargs=6, type="float", help="elements in bounding box appear in red")

parser.set_defaults(size=(600,1000),opacity=False,camera=(20,30),interact=False,label=False,filename=None,box=None)

(options, args) = parser.parse_args()


def highlight_box(pn,box,default_attr):

    """return the a dictionnary [i:h} where i is the pore index and h is a bool
    h is True if the pore coordinates are inside the provided box, False otherwise
    """

    highlight = {}

    for p in pn.graph.nodes:
        if ((pn[p]['center'][0] >= box[0] or pn[p]['center'][0] <= box[1])
            or (pn[p]['center'][1] >= box[2] or pn[p]['center'][0] <= box[3])
            or (pn[p]['center'][2] >= box[4] or pn[p]['center'][0] <= box[5])):

            highlight[p] = True

        else:
            highlight[p] = False

    return highlight

def InitRenderer(pn):

    pnviewer = pnm.PNM_3d_viewer(pn)
    pnviewer.make_geometry(pore_resolution = 8, throat_resolution = 4,render_throats=True)
    return pnviewer

def render( pnviewer,
            render=True,
            save=True,
            cmap = u'viridis',
            filename=None,
            attr='radius',
            high=None,
            low=None):


    import networkx as nx
    import numpy as np
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt

    if high is None or low is None:
        try:
            attrval = nx.get_node_attributes(pnviewer.network.graph,attr).values()
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

    pores_cm = {n:cmap(norm(r))[:-1] for (n,r) in nx.get_node_attributes(pnviewer.network.graph,attr).items()}
    throats_cm = [cmap(norm(r))[:-1] for r in nx.get_edge_attributes(pnviewer.network.graph,attr).values()]

    pnviewer.maps_colors(pores_colormap = pores_cm,throats_colormap=throats_cm,render_throats=True)

    if render:
        viewer.render(azimuth=25,elevation=30)
    if save:
        if filename is None:
            extent = pnviewer.network.graph.graph['extent']
            filename = "PNM-{0:d}n-{1:d}e-{2:.2f}x{3:.2f}x{4:.2f}.jpg".format(pnviewer.network.graph.number_of_nodes(),pnviewer.network.graph.number_of_edges(),extent[0],extent[1],extent[2])
        pnviewer.save(filename,azimuth=25,elevation=30)


def main():

    print(options)

    if options.filename is not None:
        filename = os.path.join(os.getcwd(),options.filename)
        pn = pnm.pore_network.read_ascii(filename)
        fileout = filename.replace("network","jpg")
        pnm.vtk_viewer( pn,render=options.interact,
                        save=True,
                        cmap = options.cmap,
                        filename=fileout,
                        attr=options.attr,
                        high=options.vmax,
                        low=options.vmin)

    else: #Parcours de tous les fichiers network du répertoire courant
        i = 0
        for filename in glob.glob('*.network'):
            fileout = filename.replace("network","jpg")
            print("Saving visualization of network {}".format(filename))
            if not os.path.isfile(fileout):   #Prevent erasing exiting files - they must be deleted manually
                if i == 0:
                    pn = pnm.pore_network.read_ascii(filename)
                    pnviewer = InitRenderer(pn)

                render( pnviewer,
                        render=options.interact,
                        save=True,
                        cmap = options.cmap,
                        filename=fileout,
                        attr=options.attr,
                        high=options.vmax,
                        low=options.vmin)


if __name__ == '__main__':
    main()

