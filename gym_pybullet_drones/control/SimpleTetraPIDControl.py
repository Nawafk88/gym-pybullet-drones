import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
from gym_pybullet_drones.utils.utils import nnlsRPM, quaternion_inverse, quaternion_multiply

class SimpleTetraPIDControl(BaseControl):
    """Generic PID control class without yaw control on tetra config.

    Based on https://github.com/prfraanje/quadcopter_sim.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.Tetra:
            print("[ERROR] in SimpleTetraPIDControl.__init__(), SimpleTetraPIDControl requires DroneModel.Tetra")
            exit()
        # self.P_COEFF_FOR = np.array([6.5, 6.5, 2.8])
        # self.I_COEFF_FOR = np.array([3., 3., 3.])
        # self.D_COEFF_FOR = np.array([0.3, 0.3, 0.3])
        self.KP1 = np.array([6.5, 6.5, 2.8])
        self.KP2 = np.array([0.15, 0.15, 0.15])
        self.KI = np.array([0.2, 0.2, 0.2])
        self.KD = np.array([0.003, 0.003, 0.003])
        self.KP1_alt = 1
        self.KP2_alt = 4
        self.KI_alt = 2
        self.P_COEFF_FOR = np.array([6.1, 6.1, 2.5])
        self.I_COEFF_FOR = np.array([.0001, .0001, .0001])
        self.D_COEFF_FOR = np.array([.3, .3, .3])
        self.P_COEFF_TOR = np.array([0.0003, 0.0003, 0.0003])
        self.I_COEFF_TOR = np.array([.00001, .00001, .00001])
        self.D_COEFF_TOR = np.array([.000003, .000003, .000003])
        self.MAX_ROLL_PITCH = np.pi/6
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        # self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        # self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        # self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.MAX_THRUST = 5.3*1000
        self.MAX_XY_TORQUE = 0.078
        self.MAX_Z_TORQUE = 0.078
        # self.A = np.array([ [1, 1, 1, 1], [0, 0, np.sqrt(3)/2, -np.sqrt(3)/2], [0, 1, -0.5, -0.5], [1, -1, -1, 1] ])
        self.A = np.array([ [self.KF, self.KF, self.KF, self.KF], [0, 0, (self.KF*self.L)*(np.sqrt(3)/2), -(self.KF*self.L)*(np.sqrt(3)/2)], [0, (self.KF*self.L), -0.5*(self.KF*self.L), -0.5*(self.KF*self.L)], [self.KM, -1*self.KM, -1*self.KM, self.KM] ])
        self.INV_A = np.linalg.inv(self.A)
        # self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        self.B_COEFF = np.array([1, 1, 1, 1])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.last_e_omega = np.zeros(3)
        self.integral_e_omega = np.zeros(3)
        self.last_e_vz = 0
        self.integral_e_vz = 0
    
    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_simplePIDPositionControl()` and `_simplePIDAttitudeControl()`.
        Parameters `cur_ang_vel`, `target_rpy`, `target_vel`, and `target_rpy_rates` are unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        if target_rpy[2]!=0:
            print("\n[WARNING] ctrl it", self.control_counter, "in SimpleTetraPIDControl.computeControl(), desired yaw={:.0f}deg but locked to 0. for DroneModel.HB".format(target_rpy[2]*(180/np.pi)))
        thrust = self._tetraAltitudeControl(control_timestep, cur_pos, target_pos, cur_vel, target_vel)
        rpm = self._tetraPIDAttitudeControl(control_timestep, thrust, cur_quat, target_rpy, cur_ang_vel)
        # cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, 0, 0

    ################################################################################
    
    def _tetraPIDAttitudeControl(self,
                                  control_timestep,
                                  thrust,
                                  cur_quat,
                                  target_rpy,
                                  cur_ang_vel
                                  ):
        """Simple PID attitude control (with yaw fixed to 0).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the computed the target roll, pitch, and yaw.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        # cur_rpy = p.getEulerFromQuaternion(cur_quat)
        quat_sp = p.getQuaternionFromEuler(target_rpy)
        cur_quat_inv = quaternion_inverse(cur_quat)
        e_i = quaternion_multiply(np.roll(quat_sp, 1), cur_quat_inv)
        ang_vel_sp =  np.multiply(self.KP1, np.sign(e_i[0])*e_i[1:])
        e_omega = ang_vel_sp - cur_ang_vel
        d_e_omega = (e_omega - self.last_e_omega) / control_timestep
        self.last_e_omega = e_omega
        self.integral_e_omega = self.integral_e_omega + e_omega*control_timestep
        #### PID target torques ####################################
        target_torques = np.multiply(self.KP2, e_omega) \
                         + np.multiply(self.KI, self.integral_e_omega) \
                         + np.multiply(self.KD, d_e_omega)
        return nnlsRPM(thrust=thrust,
                       x_torque=target_torques[0],
                       y_torque=target_torques[1],
                       z_torque=target_torques[2],
                       counter=self.control_counter,
                       max_thrust=self.MAX_THRUST,
                       max_xy_torque=self.MAX_XY_TORQUE,
                       max_z_torque=self.MAX_Z_TORQUE,
                       a=self.A,
                       inv_a=self.INV_A,
                       b_coeff=self.B_COEFF,
                       gui=True
                       )
    ################################################################################

    def _tetraAltitudeControl(self,
                                  control_timestep,
                                  cur_pos,
                                  target_pos,
                                  cur_vel,
                                  target_vel
                                  ):
        """Tetra Altitude controller

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.

        Returns
        -------
        float
            The target thrust along the drone z-axis.

        """
        e_z = 1*(target_pos[2] - cur_pos[2])
        print("e_z")
        print(e_z)
        vel_z_sp = self.KP1_alt*e_z
        e_vz = vel_z_sp - cur_vel[2]
        #d_e_vz = (e_vz - self.last_e_vz) / control_timestep
        #self.last_e_vz = e_vz
        self.integral_e_vz = self.integral_e_vz + e_vz*control_timestep
        #### PID target thrust ####################################
        a_com = self.KP2_alt*e_vz + self.KI_alt*self.integral_e_vz
        # a_com = 0.1
        target_thrust = 1*(1*9.8 + 1*a_com)
        return target_thrust
