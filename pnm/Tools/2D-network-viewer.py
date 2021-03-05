#!/usr/bin/python
# -*- coding: iso-8859-15

'''
Récupération des données des fichiers .network (difpnv1.6 et supérieure)
Visualisation de réseau 2D sortie PNG 


A Faire : pression moyenne dans un pore comme somme pondérée par la saturation...

Format des fichier network: 

header : Lx Ly Lz npores ncaps
pores_data
caps_data


Format pores_data: x y z radius sw pw pnw cc pout cl
x y z :    float, position du centre du pore
radius :   float, rayon du pore
sw :       float, saturation en fluide mouillant
pw, pnw :  float, pressions du fluide mouillant et non mouillant 
cc :       int, label du ganglion de fluide non mouillant
pout :     int, indique la connectivité au milieu extérieur
cl :       int, appartenance à un bord

Format caps_data : pi pj radius sw cc
pi, pj :  int, index des pores adjacents
radius :  float, rayon du capillaire
sw :      float, saturation en fluide mouillant
cc :      int, label du capillaire
intpos    (facultatif) de quel côté se trouve le fluide.
'''

import string
import re
import math
import sys
import os
import glob
from numpy import *
from optparse import OptionParser
from PIL import Image, ImageDraw, ImageColor
import colorsys

#Récupération des paramètres d'entrée

parser = OptionParser()

parser.add_option("-f", "--file", dest="filename",default=None,
                  help="FILE to process. If not present all files with file with .network extension are processed"
                  , metavar="FILE")
parser.add_option("-l", "--label", dest="label", action="store_true",help="Render labels instead of saturation")
parser.add_option("-p", "--pressure", dest="pressure", action="store_true",help="Render pressure instead of saturation")
parser.add_option("-s","--size",dest="size",help="Size of the render window (lx)",
                  metavar="lx",type="int")
parser.add_option("-c","--color",type="int",dest="points",action="append",
                  help="list of points to be highlighted")
parser.add_option("--p0",dest="p0",metavar="p0",type="float",default=1e5,
                  help="Ref. pressure")
parser.add_option("--prange",dest="prange",metavar="prange",type="float",nargs=2,default=(0,0),
                  help="Min and max pressure")
parser.add_option("--xperiodic", dest="xperiodic", action="store_true", help="Network is periodic in the x direction")
parser.add_option("--yperiodic", dest="yperiodic", action="store_true", help="Network is periodic in the y direction")
parser.add_option("--zperiodic", dest="zperiodic", action="store_true", help="Network is periodic in the z direction")
                  
parser.set_defaults(size=640,label=False,filename=None)
                  
(options, args) = parser.parse_args()
if (options.pressure) and (options.label):
  print "Argument -p et -l incompatibles ! l par défaut"
  options.pressure=False
#Routine d'ouverture d'un fichier et gestion des exceptions

def fopen(filename):
  try:
    filein=open(filename,"r")
    print "Ouverture du fichier", filename
    return filein
  except IOError:
    print "Erreur d'ouverture du fichier", filename
    sys.exit(-1)

def network_reader(header,caps,pores):
  #Récupération des données  --------------------------------------------------------------------------------------------------------------------------

#  Récupération de l'en tête
  del(header[:])
  header=filein.readline().split()
#  nbr de pores et capillaires
  npores=int(header[3])
  ncaps=int(header[4])
  
# print "Récupération des",npores,"pores"
  del(pores[:])
  for i in range(npores):
    pore_data=filein.readline().split()
    pores.append(pore_data)

# print "Récupération des",ncaps,"capillaires"
  del(caps[:])
  for i in range(ncaps):
    cap_data=filein.readline().split()
    caps.append(cap_data)

  return npores,ncaps,header,caps,pores
  
#Fin récupération des données  ----------------------------------------------------------------------------------------------------------------------

def Save(image,filename):

  fileoutname=str(filename)
  fileoutname=fileoutname.replace("network","png")
  if options.label:
    fileoutname="l-"+fileoutname
  elif options.pressure:
    fileoutname="p-"+fileoutname
  print 'Ecriture du fichier ',fileoutname,"\n"
  image.save(fileoutname,"PNG")
  
#----------------------------------------------------------------------------------------------------------------------------------------------------
def GetRes(t_size): 
  global res
  res=float(float(options.size)/float(header[0]))
  t_size=(options.size,int((res)*float(header[1])))
  return t_size,res

#----------------------------------------------------------------------------------------------------------------------------------------------------

#def Get_prange():

#----------------------------------------------------------------------------------------------------------------------------------------------------
def Net_draw(im,res):
 
  global minp
  global maxp
 
  if options.label:
    maxval=max(max(int(caps[i][4]) for i in xrange(len(caps))),max(int(pores[j][7]) for j in xrange(len(pores))))
    print "label max=",maxval
    label=[ [0,i] for i in xrange(int(maxval)) ]
    for i in xrange(ncaps):
      if int(caps[i][4])>0:label[int(caps[i][4])-1][0]+=1
    for i in xrange(npores):
      if int(pores[i][7])>0:label[int(pores[i][7])-1][0]+=1
    #Création de la table des labels afin que les composantes contenant le plus de pores aient tjrs le label le plus petit
    label.sort(reverse=True)
    for i in xrange(len(label)): 
      label[i][0]=i
    label.sort(key=lambda x:x[1])
  
  elif options.pressure:
    if options.prange==(0,0):
      maxp=max(pressure(j) for j in xrange(npores))
      minp=min(pressure(j) for j in xrange(npores))
      if minp==maxp: 
        minp=0
        maxp=1
      print minp,maxp
    else:
      maxp=options.prange[1]
      minp=options.prange[0]
      #print minp,maxp
  
  draw = ImageDraw.Draw(im)
  for i in range(ncaps):
    CL=int(pores[int(caps[i][0])-1][9])+int(pores[int(caps[i][1])-1][9])
    if (options.xperiodic and (CL & 768)==768) or (options.yperiodic and (CL & 3072)==3072) or (options.zperiodic and (CL & 12288)==12288) :
      pass
    else:
      cx=int  (res*float(pores[int(caps[i][0])-1][0]))
      cy=int  (res*float(pores[int(caps[i][0])-1][1]))
      cx2=int (res*float(pores[int(caps[i][1])-1][0]))
      cy2=int (res*float(pores[int(caps[i][1])-1][1]))
      rad=int (res*float(caps[i][2]))
      
      if rad<1: rad=1
      #draw.rectangle((cx-d,cy-d,cx2+d,cy2+d),fill=color,outline="black")
      if not (options.label or options.pressure):
        r,g,b=int((1.0-float(caps[i][3]))*255),int((1.0-float(caps[i][3]))*255), 255
      elif(options.label):
        if int(caps[i][4])>0:
          r,g,b=colorsys.hsv_to_rgb(float(label[int(caps[i][4])-1][0])/float(maxval),float(int(label[int(caps[i][4])-1][0])%2)/2+0.5,1)
          r,g,b=int(r*255),int(g*255),int(b*255)
        else:
          r,g,b=255,255,255
      elif(options.pressure):
        r,g,b=colorsys.hsv_to_rgb(0.664*(normp(int(caps[i][0])-1) + normp(int(caps[i][1])-1))/2,1,1)
        r,g,b=int(r*255),int(g*255),int(b*255)
      draw.rectangle((cx-rad,cy-rad,cx2+rad,cy2+rad),fill=(r,g,b),outline="black")
    
  for i in range(npores):
    cx=int (res*float(pores[i][0]))
    cy=int (res*float(pores[i][1]))
    rad=int (res*float(pores[i][3]))
    if rad<1: rad=1
    if not (options.label or options.pressure):
      r,g,b=int((1.0-float(pores[i][4]))*255),int((1.0-float(pores[i][4]))*255), 255
      if options.points:
        for hlabel in options.points:
          if int(hlabel)==i: r,g,b=255,0,0
    elif(options.label):
      if int(pores[i][7])>0:
        r,g,b=colorsys.hsv_to_rgb(float(label[int(pores[i][7])-1][0])/float(maxval),float(int(label[int(pores[i][7])-1][0])%2)/2+0.5,1)
        r,g,b=int(r*255),int(g*255),int(b*255)
        #print label[int(pores[i][7])-1][0],r,g,b
      else:
        r,g,b=255,255,255
    elif(options.pressure):
      r,g,b=colorsys.hsv_to_rgb(normp(i)*0.664,1,1)
      r,g,b=int(r*255),int(g*255),int(b*255)
    draw.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),fill=(r,g,b),outline="black")

#----------------------------------------------------------------------------------------------------------------------------------------------------

def pressure(i):
  #retourne la pression dans un pore  !
  if float(pores[i][4])<=0:
    return float(pores[i][6])
  elif float(pores[i][4])>=1:
    return float(pores[i][5])
  else:
    return (float(pores[i][6])*(1-float(pores[i][4]))+float(pores[i][5])*float(pores[i][4]))

def normp(i):
  #retourne la pression normalisée
  global minp
  global maxp
  #print float((pressure(i)-minp)/(maxp-minp))
  normp=float((pressure(i)-minp)/(maxp-minp))
  if normp<0:normp=0
  if normp>1:normp=1
  return normp
  

#~ initialisation
it=0
counter=0
pores=[]
caps=[]
header=[]
t_size=()
res=1.0
minp=0.
maxp=1.
p0=options.p0

if options.filename!=None:
  filein=fopen(options.filename)
  npores,ncaps,header,caps,pores=network_reader(header,caps,pores)
  t_size,res=GetRes(t_size)
  image=Image.new("RGB",t_size,"white")
  Net_draw(image,res)
  Save(image,options.filename)

else: #Parcours de tous les fichiers network du répertoire courant
  for filename in glob.glob('*.network'): 
    it+=1
    filein=fopen(filename)
    counter=counter+1
    npores,ncaps,header,caps,pores=network_reader(header,caps,pores)
    t_size,res=GetRes(t_size)
    image=Image.new("RGB",t_size,"white")
    Net_draw(image,res)
    Save(image,filename)
