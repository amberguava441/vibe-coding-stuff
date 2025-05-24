from pb import *
import heapq
import math
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d

class GlobalMap:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 1
        self.tx = 0
        self.ty = 0
        self.tdx = 0
        self.tdy = 1
        self.o = 0
        self.a = 0

        self.fx = []
        self.fy = []
        self.fl = []
        self.pl = []

        self.path = []

class LocalMap:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.dx = 0
        self.dy = 1
        self.tx = 0
        self.ty = 0
        self.tdx = 0
        self.tdy = 1

        self.ol = []
        self.lanel = []
        self.laner = []

        self.lanelx = []
        self.lanely = []
        self.lanerx = []
        self.lanery = []

        self.fx = []
        self.fy = []
        self.fl = []
        self.pl = []

        self.path = []

class location():
    def __init__(self, map, j=None, w=None):
        self.map = map
        if j is None or w is None:
            j = map.ji
            w = map.wi
        self.setjw(j, w)

    def ifobstacle(self): #0为道路，1非道路，2为校外区域
        if self.map.map[self.r, self.c] == 0:
            self.safe = True
        else:
            self.safe = False

    def setrc(self, r, c):
        self.r = r
        self.c = c
        self.j, self.w = self.rctojw(r, c)
        self.x, self.y = self.rctoxy(r, c)
        self.ifobstacle()

    def setjw(self, j, w):
        self.j = j
        self.w = w
        self.r, self.c = self.jwtorc(j, w)
        self.x, self.y = self.jwtoxy(j, w)
        self.ifobstacle()

    def setxy(self, x, y):
        self.x = x
        self.y = y
        self.r, self.c = self.xytorc(x, y)
        self.j, self.w = self.xytojw(x, y)
        self.ifobstacle()

    def jwtorc(self, j, w):
        return jwtorc(self.map, j, w)

    def xytorc(self, x, y):
        return xytorc(self.map, x, y)

    def rctojw(self, r, c):
        return rctojw(self.map, r, c)

    def rctoxy(self, r, c):
        return rctoxy(self.map, r, c)

    def jwtoxy(self, j, w):
        return jwtoxy(self.map, j, w)

    def xytojw(self, x, y):
        return xytojw(self.map, x, y)

class Astarplanner():
    def __init__(self, map):
        self.map = map
        self.path = []
        self.pathxy = []
        self.pathjw = []
        self.setsafedistance(safedistance=10, maxcost=10)
        self.direction = [(-1, -1, 1.414), (-1, 0, 1), (-1, 1, 1.414), (0, -1, 1), (0, 1, 1), (1, -1, 1.414), (1, 0, 1), (1, 1, 1.414)]

    def setsafedistance(self, safedistance = 0, maxcost = 10):
        safemap = np.zeros((self.map.xl, self.map.yl))
        if safedistance!=0 and maxcost!=0:
            safedistance = safedistance/self.map.size
            map = self.map.map == 0
            distance = distance_transform_edt(map)
            safemap[distance==0] = np.inf
            distance[distance==0] = np.inf
            safemap[distance <= safedistance] = maxcost / (distance[distance <= safedistance])
        self.map.safemap = safemap

    def plan(self, xy, txy):
        self.x, self.y = xy
        self.tx, self.ty = txy
        self.path = []
        self.pathxy = []
        self.pathjw = []
        if self.inner(self.x,self.y) and self.inner(self.tx,self.ty) and self.safe(self.x,self.y) and self.safe(self.tx,self.ty):
            self.path = self.Astar(self.x, self.y, self.tx, self.ty)
            self.pathxy = [self.map.mapxy[x, y] for (x, y) in self.path]
            self.pathjw = [self.map.mapjw[x, y] for (x, y) in self.path]
            self.pathxy = [(xy.x, xy.y) for xy in self.pathxy]
            self.pathjw = [(xy.x, xy.y) for xy in self.pathjw]
        return True if self.path else False

    def writepath(self):
        map = np.ones((self.map.xl, self.map.yl, 3), dtype=np.uint8) * 255
        map[self.map.map==1] = [0, 255, 0]
        map[self.map.map==2] = [0, 0, 0]
        if self.path:
            for (x, y) in self.path:
                map[x, y] = [0, 0, 255]

        map[self.x, self.y] = [255, 0, 255]
        map[self.tx, self.ty] = [255, 255, 0]

        cv.imwrite("path.png", map.swapaxes(0, 1)[::-1, :, :])

    def getpath(self):
        return self.path, self.pathxy, self.pathjw

    def safe(self, x, y):
        return 0==self.map.map[x, y]

    def inner(self, x, y):
        return 0<=x<self.map.xl and 0<=y<self.map.yl

    def heuristic(self, xy, txy):
        return ((xy[0]-txy[0])**2 + (xy[1]-txy[1])**2)**0.5

    def Astar(self, x, y, tx, ty):

        path = []

        g = {(x, y): 0}
        f = {(x, y): self.heuristic((x, y), (tx, ty))}

        openset = []
        parent = {}

        heapq.heappush(openset, (f[(x, y)], 0, x, y))

        while openset:
            _, gn, xn, yn = heapq.heappop(openset)

            if xn == tx and yn == ty:
                while (xn, yn) in parent:
                    path.append((xn, yn))
                    xn, yn = parent[(xn, yn)]
                path.append((x, y))
                path.reverse()
                break

            for dx, dy, dl in self.direction:
                xc, yc = xn + dx, yn + dy

                if self.inner(xc, yc) and self.safe(xc, yc):
                    
                    gc = gn + dl + self.map.safemap[xc, yc]

                    if (xc, yc) not in g or gc < g[(xc, yc)]:
                        parent[(xc, yc)] = (xn, yn)
                        g[(xc, yc)] = gc
                        f[(xc, yc)] = gc + self.heuristic((xc, yc), (tx, ty))
                        heapq.heappush(openset, (f[(xc, yc)], gc, xc, yc))

        return path

class VFHplanner():
    def __init__(self, map, safedistance = 0.5, pathinterval = 0.01, maxk = 0.167):
        self.map = map

        self.globalmap = GlobalMap()
        self.localmap = LocalMap()

        self.safedistance = safedistance
        self.pathinterval = pathinterval
        self.maxk = maxk

    def setglobalmap(self, xy, dxy, txy, tdxy):
        self.globalmap.x, self.globalmap.y = xy
        self.globalmap.tx, self.globalmap.ty = txy

        dx, dy = dxy
        dxyl = (dx**2 + dy**2)**0.5
        self.globalmap.dx = dx / dxyl
        self.globalmap.dy = dy / dxyl

        tdx, tdy = tdxy
        tdxyl = (tdx**2 + tdy**2)**0.5
        self.globalmap.tdx = tdx / tdxyl
        self.globalmap.tdy = tdy / tdxyl

        self.globalmap.o = self.getorientation(self.globalmap.dx, self.globalmap.dy)
        self.globalmap.a = math.atan2(self.globalmap.dy, self.globalmap.dx)-np.pi/2

    def setScenario(self, xy, dxy, txy, tdxy, lane, ol):
        self.setglobalmap(xy, dxy, txy, tdxy)

        self.getlocalmap(lane, ol)

        return 0==self.check()

    def getlocalmap(self, lane, ol):
        a = self.globalmap.a
        tx = self.globalmap.tx-self.globalmap.x
        ty = self.globalmap.ty-self.globalmap.y

        self.localmap.tx, self.localmap.ty = self.rotation(a, tx, ty)
        self.localmap.tdx, self.localmap.tdy = self.rotation(a, self.globalmap.tdx, self.globalmap.tdy)

        self.setlane(lane)
        self.setol(ol)

    def setol(self, ol):
        self.localmap.ol = []
        if ol and all(isinstance(r, list) and len(r) == 7 for r in ol):
            for _, x, y, z, w, l, h in ol:
                self.localmap.ol.append([(x, y, z), (w, l, h)])

    def setlane(self, lane):
        self.localmap.lanel = []
        self.localmap.laner = []
        if lane and 2==len(lane):
            lanel = [xy for xy in lane[0] if xy != (None, None)]
            laner = [xy for xy in lane[1] if xy != (None, None)]
            if lanel and laner:
                self.localmap.lanel = lanel
                self.localmap.laner = laner

    def rotation(self, a, x, y):
        return x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)

    def check(self):
        if self.localmap.lanel and self.localmap.laner and self.localmap.tx and self.localmap.ty and self.localmap.tdx and self.localmap.tdy:
            return 0
        return 1

    def plan(self, xy, dxy, txy, tdxy, lane, ol):
        self.localmap.fx = []
        self.localmap.fy = []
        self.localmap.fl = []
        self.localmap.pl = []
        self.globalmap.fx = []
        self.globalmap.fy = []
        self.globalmap.fl = []
        self.globalmap.pl = []
        if self.setScenario(xy, dxy, txy, tdxy, lane, ol):
            self.localmap.fx, self.localmap.fy, self.localmap.fl, self.localmap.pl =  self.VFH(
                                        self.localmap.x, self.localmap.y, self.localmap.dx, self.localmap.dy,
                                        self.localmap.tx, self.localmap.ty, self.localmap.tdx, self.localmap.tdy,
                                        self.localmap.ol, self.localmap.lanel, self.localmap.laner,
                                        self.maxk, self.safedistance)
            if self.localmap.pl:
                return True
        return False
    
    def transform(self, x, y):
        a = -self.globalmap.a
        for rx, ry in zip(self.localmap.fx, self.localmap.fy):
            dx, dy = self.rotation(a, rx, ry)
            self.globalmap.fx.append(dx)
            self.globalmap.fy.append(dy)
        self.globalmap.fl = self.localmap.fl
        self.globalmap.pl = self.localmap.pl
        self.globalmap.fx[0] += x
        self.globalmap.fy[0] += y
            
    def getsample(self, fx, fy, fl, pl):
        path = []
        if fx and fy and fl and pl:
            pc = np.ceil(pl/self.pathinterval).astype(int)
            px = np.polyval(fx[::-1], np.linspace(0, fl, pc))
            py = np.polyval(fy[::-1], np.linspace(0, fl, pc))
            for x, y in zip(px, py):
                path.append((x,y))
        return path

    def getorientation(self, dx, dy):
        self.o = 90 - math.degrees(math.atan2(dy, dx)) 

        if self.o < 0:
            self.o += 360

        return self.o

    def writepath(self, suffix = []):
        map = np.ones((self.map.xl, self.map.yl, 3), dtype=np.uint8) * 255
        map[self.map.map==1] = [0, 255, 0]
        map[self.map.map==2] = [0, 0, 0]

        bx = self.globalmap.x
        by = self.globalmap.y
        a = -self.globalmap.a

        for rx, ry in self.localmap.lanel:
            dx, dy = self.rotation(a, rx, ry)
            x = bx+dx
            y = by+dy
            r, c = xytorc(self.map,x,y)
            map[r,c] = [255, 0, 0]

        for rx, ry in self.localmap.laner:
            dx, dy = self.rotation(a, rx, ry)
            x = bx+dx
            y = by+dy
            r, c = xytorc(self.map,x,y)
            map[r,c] = [255, 0, 0]

        for xyz, wlh in self.localmap.ol:
            cx, cy, _ = xyz
            w, l, _ = wlh
            w = w
            l = l
            rdx = np.concatenate((np.arange(0, -w, -1), np.arange(1, w, 1)))
            rdy = np.concatenate((np.arange(0, -l, -1), np.arange(1, l, 1)))
            for rx in rdx:
                for ry in rdy:
                    dx, dy = self.rotation(a, cx+rx, cy+ry)
                    x = bx+dx
                    y = by+dy
                    r, c = xytorc(self.map,x,y)
                    map[r,c] = [255, 0, 0]
            
        if self.localmap.path:
            for rx, ry in self.localmap.path:
                dx, dy = self.rotation(a, rx, ry)
                x = bx+dx
                y = by+dy
                r, c = xytorc(self.map,x,y)
                map[r, c] = [0, 0, 255]

        # if self.globalmap.path:
        #     for dx, dy in self.globalmap.path:
        #         x = bx+dx-self.globalmap.path[0][0]
        #         y = by+dy-self.globalmap.path[0][1]
        #         r, c = xytorc(self.map,x,y)
        #         map[r, c] = [0, 0, 255]

        dx, dy = self.rotation(a, self.localmap.x, self.localmap.y)
        x = bx+dx
        y = by+dy
        r, c = xytorc(self.map,x,y)
        map[r, c] = [255, 0, 255]

        dx, dy = self.rotation(a, self.localmap.tx, self.localmap.ty)
        x = bx+dx
        y = by+dy
        r, c = xytorc(self.map,x,y)
        map[r, c] = [255, 255, 0]

        if suffix:
            cv.imwrite(f"trajectory{suffix}.png", map.swapaxes(0, 1)[::-1, :, :])
        else:
            cv.imwrite("trajectory.png", map.swapaxes(0, 1)[::-1, :, :])

    def getpath(self):
        self.localmap.path = self.getsample(self.localmap.fx, self.localmap.fy, self.localmap.fl, self.localmap.pl)
        self.globalmap.path = self.getsample(self.globalmap.fx, self.globalmap.fy, self.globalmap.fl, self.globalmap.pl)
        return (self.localmap.fx, self.localmap.fy, self.localmap.fl, self.localmap.pl), (self.globalmap.fx, self.globalmap.fy, self.globalmap.fl, self.globalmap.pl), self.localmap.path, self.globalmap.path

    def VFH(self, x, y, dx, dy, tx, ty, tdx, tdy, ol, lanel, laner, maxk, sd):

        lanel = np.array(lanel)
        laner = np.array(laner)

        lanel = lanel[lanel[:, 1].argsort()]
        laner = laner[laner[:, 1].argsort()]

        lanelx = lanel[:, 0]
        lanely = lanel[:, 1]
        lanerx = laner[:, 0]
        lanery = laner[:, 1]

        lanel = interp1d(lanely, lanelx, kind='linear', fill_value='extrapolate')
        laner = interp1d(lanery, lanerx, kind='linear', fill_value='extrapolate')

        ia = 0.1
        maxa = 0.8

        fl = max(np.hypot(tx - x, ty - y), 1.0)

        fx = self.solvequinticcoefficient(x, dx, 0.0, tx, tdx, 0.0, fl)
        fy = self.solvequinticcoefficient(y, dy, 0.0, ty, tdy, 0.0, fl)
        if self.ifsafe(fx, fy, fl, lanely, lanery, ol, lanel, laner, sd) and self.ifsmooth(fx, fy, fl, maxk):
            return fx, fy, fl, self.getpathlength(fx, fy, fl)

        bestfx = []
        bestfy = []
        bestpl = float('inf')

        a = ia
        while a <= maxa:
            for lateral in [-1.0, 1.0]:
                fx = self.solvequinticcoefficient(x, dx, lateral * a, tx, tdx, 0.0, fl)
                fy = self.solvequinticcoefficient(y, dy, 0.0, ty, tdy, 0.0, fl)
                if self.ifsafe(fx, fy, fl, lanely, lanery, ol, lanel, laner, sd) and self.ifsmooth(fx, fy, fl, maxk):
                    pl = self.getpathlength(fx, fy, fl)
                    if pl < bestpl:
                        bestpl = pl
                        bestfx = fx
                        bestfy = fy
                    break
            a += ia

        if bestfx and bestfy and bestpl:
            return bestfx, bestfy, fl, bestpl

        while ia <= maxa:
            fl *= 1.5
            ia *= 2
            
            a = ia
            while a <= maxa:
                for lateral in [-1.0, 1.0]:
                    fx = self.solvequinticcoefficient(x, dx, lateral * a, tx, tdx, 0.0, fl)
                    fy = self.solvequinticcoefficient(y, dy, 0.0, ty, tdy, 0.0, fl)
                    if self.ifsafe(fx, fy, fl, lanely, lanery, ol, lanel, laner, sd) and self.ifsmooth(fx, fy, fl, maxk):
                        return fx, fy, fl, self.getpathlength(fx, fy, fl)
                a += ia

        return [], [], [], []
    
    def solvequinticcoefficient(self, p, v, a, tp, tv, ta, T):
        A0 = p
        A1 = v
        A2 = a / 2.0  # a 是初始加速度，满足 x''(0) = 2*A2 = a
        # 建立关于 A3, A4, A5 的线性方程组:
        T2, T3 = T*T, T**3
        T4, T5 = T3*T, T3*T2
        Am = np.array([[T3,    T4,    T5],
                              [3*T2,  4*T3,  5*T4],
                              [6*T,  12*T2, 20*T3]])
        Bv = np.array([
            tp - (A0 + A1*T + A2*T*T),        # x(T) - 已知部分 = 0
            tv - (A1 + 2*A2*T),               # x'(T) - 已知部分 = 0
            ta - (2*A2)                       # x''(T) - 已知部分 = 0
        ])
        # 解方程 Am * [A3, A4, A5]^T = Bv
        X = np.linalg.solve(Am, Bv)
        A3, A4, A5 = X.tolist()
        return [A0, A1, A2, A3, A4, A5]

    def ifsafe(self, fx, fy, fl, lanely, lanery, ol, lanel, laner, sd):
        for t in np.linspace(0, fl, 100):
            x = fx[0] + fx[1]*t + fx[2]*t**2 + fx[3]*t**3 + fx[4]*t**4 + fx[5]*t**5
            y = fy[0] + fy[1]*t + fy[2]*t**2 + fy[3]*t**3 + fy[4]*t**4 + fy[5]*t**5
            if y < lanely[0] or y > lanely[-1] and y < lanery[0] or y > lanery[-1]:
                return False
            if x < lanel(y)+sd or x > laner(y)-sd:
                return False

            for oxyz, owlh in ol:
                ox, oy, _ = oxyz
                w, l, _ = owlh
                if abs(x - ox) <= w/2.0+sd and abs(y - oy) <= l/2.0+sd:
                    return False
        return True

    def ifsmooth(self, fx, fy, fl, allowedk):
        t = np.linspace(0, fl, 100)

        dfx = np.poly1d([5*fx[5], 4*fx[4], 3*fx[3], 2*fx[2], fx[1]])
        dfy = np.poly1d([5*fy[5], 4*fy[4], 3*fy[3], 2*fy[2], fy[1]])
        ddfx = np.poly1d([20*fx[5], 12*fx[4], 6*fx[3], 2*fx[2]])
        ddfy = np.poly1d([20*fy[5], 12*fy[4], 6*fy[3], 2*fy[2]])
        
        dfx = dfx(t)
        dfy = dfy(t)
        ddfx = ddfx(t)
        ddfy = ddfy(t)
        
        numerator = np.abs(dfx * ddfx - dfy * ddfy)
        denominator = (dfx**2 + dfy**2) ** 1.5

        k = np.where(denominator > 1e-10, numerator / denominator, 0)

        maxk = np.max(k)

        return maxk<allowedk

    def getpathlength(self, fx, fy, fl):
        pl = 0.0
        xp = fx[0]
        yp = fy[0]
        for t in np.linspace(0, fl, 100)[1:]:
            xn = fx[0] + fx[1]*t + fx[2]*t**2 + fx[3]*t**3 + fx[4]*t**4 + fx[5]*t**5
            yn = fy[0] + fy[1]*t + fy[2]*t**2 + fy[3]*t**3 + fy[4]*t**4 + fy[5]*t**5
            dx = xn - xp
            dy = yn - yp
            pl += np.hypot(dx, dy)
            xp, yp = xn, yn
        return pl
