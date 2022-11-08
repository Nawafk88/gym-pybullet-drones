"""General use functions.
"""
import time
import argparse
import numpy as np
from scipy.optimize import nnls

################################################################################

def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")

################################################################################

def nnlsRPM(thrust,
            x_torque,
            y_torque,
            z_torque,
            counter,
            max_thrust,
            max_xy_torque,
            max_z_torque,
            a,
            inv_a,
            b_coeff,
            gui=False
            ):
    """Non-negative Least Squares (NNLS) RPMs from desired thrust and torques.

    This function uses the NNLS implementation in `scipy.optimize`.

    Parameters
    ----------
    thrust : float
        Desired thrust along the drone's z-axis.
    x_torque : float
        Desired drone's x-axis torque.
    y_torque : float
        Desired drone's y-axis torque.
    z_torque : float
        Desired drone's z-axis torque.
    counter : int
        Simulation or control iteration, only used for printouts.
    max_thrust : float
        Maximum thrust of the quadcopter.
    max_xy_torque : float
        Maximum torque around the x and y axes of the quadcopter.
    max_z_torque : float
        Maximum torque around the z axis of the quadcopter.
    a : ndarray
        (4, 4)-shaped array of floats containing the motors configuration.
    inv_a : ndarray
        (4, 4)-shaped array of floats, inverse of a.
    b_coeff : ndarray
        (4,1)-shaped array of floats containing the coefficients to re-scale thrust and torques. 
    gui : boolean, optional
        Whether a GUI is active or not, only used for printouts.

    Returns
    -------
    ndarray
        (4,)-shaped array of ints containing the desired RPMs of each propeller.

    """
    #### Check the feasibility of thrust and torques ###########
    if gui and thrust < 0 or thrust > max_thrust:
        print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible thrust {:.2f} outside range [0, {:.2f}]".format(thrust, max_thrust))
    if gui and np.abs(x_torque) > max_xy_torque:
        print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible roll torque {:.2f} outside range [{:.2f}, {:.2f}]".format(x_torque, -max_xy_torque, max_xy_torque))
    if gui and np.abs(y_torque) > max_xy_torque:
        print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible pitch torque {:.2f} outside range [{:.2f}, {:.2f}]".format(y_torque, -max_xy_torque, max_xy_torque))
    if gui and np.abs(z_torque) > max_z_torque:
        print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible yaw torque {:.2f} outside range [{:.2f}, {:.2f}]".format(z_torque, -max_z_torque, max_z_torque))
    test = [thrust, x_torque, y_torque, z_torque]
    # print("this is test")
    # print(test)
    B = np.multiply(np.array([thrust, x_torque, y_torque, z_torque]), b_coeff)
    # sq_rpm = np.dot(inv_a, B)
    sq_rpm = test
    #### NNLS if any of the desired ang vel is negative ########
    if True:
        sol, res = nnls(a,
                        B,
                        maxiter=3*a.shape[1]
                        )
        if gui:
            print("[WARNING] iter", counter, "in utils.nnlsRPM(), unfeasible squared rotor speeds, using NNLS")
            print("Negative sq. rotor speeds:\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sq_rpm[0], sq_rpm[1], sq_rpm[2], sq_rpm[3]),
                   "\t\tNormalized: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sq_rpm[0]/np.linalg.norm(sq_rpm), sq_rpm[1]/np.linalg.norm(sq_rpm), sq_rpm[2]/np.linalg.norm(sq_rpm), sq_rpm[3]/np.linalg.norm(sq_rpm)))
            print("NNLS:\t\t\t\t [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sol[0], sol[1], sol[2], sol[3]),
                  "\t\t\tNormalized: [{:.2f}, {:.2f}, {:.2f}, {:.2f}]".format(sol[0]/np.linalg.norm(sol), sol[1]/np.linalg.norm(sol), sol[2]/np.linalg.norm(sol), sol[3]/np.linalg.norm(sol)),
                  "\t\tResidual: {:.2f}".format(res))
        sq_rpm = sol
    return np.sqrt(sq_rpm)
    
################################################################################

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.
    
    Output
      :return qw, qx, qy, qz: The orientation in quaternion [w,x,y,z] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qw, qx, qy, qz]
  
################################################################################

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return roll_x, pitch_y, yaw_z # in radians

################################################################################

def quaternion_inverse(quaternion):
    """Return inverse of quaternion.
        >>> q0 = random_quaternion()
        >>> q1 = quaternion_inverse(q0)
        >>> numpy.allclose(quaternion_multiply(q0, q1), [1, 0, 0, 0])
        True
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    np.negative(q[1:], q[1:])
    return q / np.dot(q, q)

################################################################################

def quaternion_multiply(quaternion1, quaternion0):
  """Return multiplication of two quaternions.

    >>> q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    >>> numpy.allclose(q, [28, -44, -14, 48])
    True

    """
  w0, x0, y0, z0 = quaternion0
  w1, x1, y1, z1 = quaternion1
  return np.array([
      -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0, x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0, x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
  ],
                     dtype=np.float64)
