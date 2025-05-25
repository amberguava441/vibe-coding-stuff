from pb import *
from pf import *

class planner():
    def __init__(self):
        self.map = creatMap()
        self.here = location(self.map)
        self.next = location(self.map)
        self.goal = location(self.map)
        self.globalplanner = Astarplanner(self.map)
        self.localplanner = VFHplanner(self.map)

        self.gp = []
        self.gt = []

        self.pl = 20

    def setOrigin(self, j, w):
        self.here.setjw(j, w)

    def setDestination(self, j, w): 
        self.goal.setjw(j, w)
        self.globalplanner.plan((self.here.r,self.here.c),(self.goal.r, self.goal.c))
        _, self.gp, _ = self.globalplanner.getpath()

    def getpath(self, xy, dxy, jw, o, lane, ol):
        self.x, self.y = xy[0], xy[1]
        self.dx, self.dy = dxy[0], dxy[1]
        self.o = o
        self.a = math.atan2(self.dy, self.dx)-np.pi/2

        self.here.setjw(jw[0],jw[1])

        if self.gp:

            self.updategp()
            
            if self.localplanner.plan((self.here.x, self.here.y), (self.dx, self.dy), (self.next.x,self.next.y), (self.tdx, self.tdy), lane, ol):
                self.localplanner.transform(self.x, self.y)
            _, self.gt, _, _ = self.localplanner.getpath()
            
            return self.gt

        return []
    
    def updategp(self):
        while len(self.gp)>1 and ((self.here.x-self.gp[0][0])**2 + (self.here.y-self.gp[0][1])**2)<self.pl**2:
            self.xl, self.yl = self.gp.pop(0)
        self.next.setxy(self.gp[0][0],self.gp[0][1])
        n = 1 if len(self.gp)>1 else 0
        self.tdx = self.gp[n][0]-self.xl
        self.tdy = self.gp[n][1]-self.yl

# p = planner()

# p.setOrigin(126.62824785,45.73069956)
# p.setDestination(126.62565295,45.72713751)
# path = p.getpath((37,-27), (-4,1), (126.62824785,45.73069956), 284.0362434679265, [], [])
