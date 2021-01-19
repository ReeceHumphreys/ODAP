
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.transform as trans

def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
        scatters[i].set_color
    return scatters

def generate_visualization(data, title):

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.set_xlim3d(-6378.0-1000, 6378.0+1000)
    ax.set_xlabel('X (km)')

    ax.set_ylim3d([-6378.0-1000, 6378.0+1000])
    ax.set_ylabel('Y (km)')

    ax.set_zlim3d([-6378.0-1000, 6378.0+1000])
    ax.set_zlabel('Z (km)')

    # Plot central body
    earth_radius = 3378.0 #km
    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    _x = earth_radius*np.cos(_u)*np.sin(_v)
    _y = earth_radius*np.sin(_u)*np.sin(_v)
    _z = earth_radius*np.cos(_v)

    earth_tilt = np.radians(23.5) # rAD
    rot_axis = np.array([0,0,1])
    rot_vec = earth_tilt * rot_axis
    rotation = trans.Rotation.from_rotvec(rot_vec)

    # _x_r = rotation.apply(_x)
    # _y_r = rotation.apply(_y)
    # _z_r = rotation.apply(_z)

    ax.plot_surface(_x, _y, _z, cmap=cm.coolwarm)


    # Initialize scatters
    scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]

    # # Number of iterations
    iterations = len(data)
    ani = FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                           interval=200, blit=False, repeat=False)

    writervideo = FFMpegWriter(fps=5)
    name =  title + ".mp4"
    ani.save(name, writer=writervideo)
