import cv2 as cv
import numpy as np
import os

class xy:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class mapdata:
    def __init__(self, map=None, mapxy=None, mapjw=None, size=None, sizejx=None, sizewy=None, xi=None, yi=None, ji=None, wi=None, xl=None, yl=None):
        self.map = map
        self.mapxy = mapxy
        self.mapjw = mapjw
        self.size = size
        self.sizejx = sizejx
        self.sizewy = sizewy
        self.xi = xi
        self.yi = yi
        self.ji = ji
        self.wi = wi
        self.xl = xl
        self.yl = yl

#def readmap(picture='map.png'):
def readmap(picture='map_null.png'):
    if os.path.exists(picture):
        picture = cv.imread(picture).swapaxes(0, 1)[:, ::-1, :]
        map = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
        map[:] = 0
        map[picture[:,:,2]>100] = 2
        map[picture[:,:,0]>100] = 1
        
        return map
    return None

def setdata(map,size,sizejx,sizewy,r,c,x,y,j,w):
    if map.map is not None:
        map.size = size
        map.sizejx = sizejx
        map.sizewy = sizewy
        map.xi = x-r*size
        map.yi = y-c*size
        map.ji = j-r*sizejx
        map.wi = w-c*sizewy

        map.xl, map.yl = map.map.shape

    return map

def getmap(map):
    if map.size is not None:
        mapxy = np.empty((map.xl, map.yl), dtype=object)
        mapjw = np.empty((map.xl, map.yl), dtype=object)
        for r in range(map.xl):
            for c in range(map.yl):
                mapxy[r, c] = xy(map.xi+r*map.size,   map.yi+c*map.size)
                mapjw[r, c] = xy(map.ji+r*map.sizejx, map.wi+c*map.sizewy)
        map.mapxy = mapxy
        map.mapjw = mapjw
    return map

def creatMap(size=0.42112,sizejx=0.0000053829375,sizewy=3.743409629044988e-6,r=743,c=1101,x=0,y=0,j=126.62781246,w=45.72934540):#size=0.42112,sizejx=0.0000053229375,sizewy=3.743409629044988e-6,r=764,c=1101,x=0,y=0,j=126.62781387,w=45.72934448
    map = mapdata(readmap())
    if map.map is not None:
        map = setdata(map,size,sizejx,sizewy,r,c,x,y,j,w)
        map = getmap(map)
    return map

def jwtorc(map, j, w):
    r = (j-map.ji)/map.sizejx
    c = (w-map.wi)/map.sizewy
    return np.round(r).astype(int), np.round(c).astype(int)

def xytorc(map, x, y):
    r = (x-map.xi)/map.size
    c = (y-map.yi)/map.size
    return np.round(r).astype(int), np.round(c).astype(int)

def rctojw(map, r, c):
    j = map.ji+map.sizejx*r
    w = map.wi+map.sizewy*c
    return j, w

def rctoxy(map, r, c):
    x = map.xi+map.size*r
    y = map.yi+map.size*c
    return x, y

def jwtoxy(map, j, w):
    x = map.xi+map.size*(j-map.ji)/map.sizejx
    y = map.yi+map.size*(w-map.wi)/map.sizewy
    return x, y

def xytojw(map, x, y):
    j = map.ji+map.sizejx*(x-map.xi)/map.size
    w = map.wi+map.sizewy*(y-map.yi)/map.size
    return j, w
