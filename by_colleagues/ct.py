import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getF(L, K, R, T):

    K = np.linalg.inv(K)

    xyz = []

    for x, y, z in L:

        p = np.array([x, y, 1])

        p = np.dot(K, p)

        x = p[0] * z
        y = p[1] * z

        p = np.dot(R.T, np.array([x, y, z])) - R.T @ T

        rx, ry, rz = p

        xyz.append((rx,ry,rz))

        #print(f"({rx:.2f}, {ry:.2f}, {rz:.2f})")

    xyz = [(point[0], point[1], 0) for point in xyz if len(point) >= 2]

    return xyz

    
    # x_vals, y_vals, z_vals = zip(*xyz)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(x_vals, y_vals, z_vals)

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()