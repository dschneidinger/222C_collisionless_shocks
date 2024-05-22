# Integrate over what is constant position in the transformed frame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy

def line_integrate(q_f: list, t0: float, t1: float, x0: float, v: float):


    # Integrate along a line with slope v, starting at some specified x position, and then normalize by that time interval 
    soln = scipy.integrate.quad(lambda t: 1/(t1-t0)*q_f([t, v*(t-t0)+x0]),t0,t1)

    # scipy.integrate returns the result and the error of numerical integration
    return soln[0], soln[1]

def box_integrate(q: list, xmin, xmax, tmin, tmax, v, plot_flag = True, q_0 = None) -> np.array:
    t_axis = q[3]
    x_axis = q[4]

    # It is necessary to interpolate your quantity in order to integrate along this path without too much headache
    q_f = scipy.interpolate.RegularGridInterpolator([t_axis,x_axis], q[0])
    dx = q[2]

    xmin_index = int(xmin/dx)
    xmax_index = int(xmax/dx)

    # Perform line integrations for successive x values in desired box
    solution = []
    for x in x_axis[xmin_index:xmax_index]: 
        solution.append(line_integrate(q_f, tmin, tmax, x, v)[0])

    # Immediately show a plot to verify, if desired
    if plot_flag: 
        plt.plot(x_axis[xmin_index:xmax_index], solution)
        if q_0 != None:
            plt.xlabel(f'{q_0.AXIS1_NAME} [${q_0.AXIS1_UNITS}$]')
            plt.ylabel(f'integrated {q_0.DATA_NAME} density')
            plt.title(f'integrated along lines of v = {v}')

    return np.array(solution)

def illustrate_box(q: list, xmin, xmax, tmin, tmax, v, q_0 = None) -> None:
    fig, ax = plt.subplots()
    # Make a heatmap of quantity in t,x space (sorry Paulo)
    plt.imshow(np.transpose(q[0]), origin='lower', extent=[q[3][0], q[3][-1], q[4][0], q[4][-1]], aspect='auto')

    # If vysxd.get_data timeshot is supplied, use this to label axes
    if (q_0 != None):
        plt.ylabel(f"{q_0.AXIS1_NAME} [${q_0.AXIS1_UNITS}$]")
        plt.xlabel(f"Time [${q_0.TIME_UNITS}$]")
        plt.colorbar(label=q_0.DATA_NAME)

    # Draw a parallelogram to illustrate entire region that is being integrated
    vertices = [[tmin,xmin],[tmin,xmax],[tmax,xmax+v*(tmax-tmin)],[tmax,xmin+v*(tmax-tmin)]]
    box = patches.Polygon(vertices, ls = '--', fill =False, color = 'red')
    ax.add_patch(box)

    # Draw arrows to illustrate which lines are being integrated over
    arrow = patches.Arrow(tmin,xmax,tmax-tmin,v*(tmax-tmin), color = 'red',width=1, ls = '--')
    ax.add_patch(arrow)
    arrow = patches.Arrow(tmin,xmin,tmax-tmin,v*(tmax-tmin), color = 'red',width=1, ls = '--')
    ax.add_patch(arrow)
    arrow = patches.Arrow(tmin,(xmax+xmin)/2,tmax-tmin,v*(tmax-tmin), color = 'red',width=1, ls = '--')
    ax.add_patch(arrow)
    
    # Need to expand limits of ax or drawings will be cut off
    ax.set_ylim(q[4][0],q[4][-1])
    ax.set_xlim(q[3][0],q[3][-1])
    plt.show()

def field_transform(v: float, e_: np.array, b_: np.array):
    gamma = 1/np.sqrt(1-v**2)

    e1 = e_[0]
    b1 = b_[0]
    # I flipped the signs...
    e2 = gamma*(e_[1] - v*b_[2])
    b2 = gamma*(b_[1] + v*e_[2])

    e3 = gamma*(e_[2] + v*b_[1])
    b3 = gamma*(b_[2] - v*e_[1])
    return [e1,e2,e3], [b1,b2,b3]

def plot_quantity(q: np.array, xmin, xmax, v, x, dx, q_0):
    xmin_index = int(xmin//dx)
    xmax_index = int(xmax/dx)
    plt.plot(x[xmin_index:xmax_index], q)
    plt.xlabel(f'{q_0.AXIS1_NAME} [${q_0.AXIS1_UNITS}$]')
    plt.ylabel(f'integrated {q_0.DATA_NAME}')
    plt.title(f'integrated along lines of $v_s = {v}$')