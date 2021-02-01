
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
    
    img = plt.imread('blue_marble.jpg')

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
    # define a grid matching the map size, subsample along with pixels
    theta = np.linspace(0, np.pi, img.shape[0])
    phi = np.linspace(0, 2*np.pi, img.shape[1])

    count = 180 # keep 180 points along theta and phi
    theta_inds = np.linspace(0, img.shape[0] - 1, count).round().astype(int)
    phi_inds = np.linspace(0, img.shape[1] - 1, count).round().astype(int)
    theta = theta[theta_inds]
    phi = phi[phi_inds]
    img = img[np.ix_(theta_inds, phi_inds)]

    theta,phi = np.meshgrid(theta, phi)
    R = earth_radius
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    
    ax.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1)

    # Initialize scatters
    scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]

    # # Number of iterations
    iterations = len(data)
    ani = FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                           interval=200, blit=False, repeat=False)

    writervideo = FFMpegWriter(fps=5)
    name =  title + ".mp4"
    ani.save(name, writer=writervideo)
