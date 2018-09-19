from pid import PID
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # Use PID controller to predict throttle value
        self.pid_controller = PID(0.1, 0.1, 0.1, 0., 1.)

        # Use Yaw controller to predict steering angle
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        self.last_time = rospy.get_time()

    def control(self, dbw_enabled, current_vel, linear_vel, angular_vel):
        rospy.logwarn('DWB enabled: {0}'.format(dbw_enabled))
        rospy.logwarn('Current velocity: {0}'.format(current_vel))
        rospy.logwarn('Target velocity: {0}'.format(linear_vel))
        rospy.logwarn('Target angular velocity: {0}'.format(angular_vel))

        # Reset PID when DBW is disable
        if not dbw_enabled:
            self.pid_controller.reset()
            return None

        # Compute current error and sample time for PID
        error = current_vel - linear_vel
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time

        # Retrieve throttle from PID controller
        throttle = self.pid_controller.step(error, sample_time)
        brake = 0
        steering = 0

        self.last_time = current_time

        rospy.logwarn('Throttle: {0}'.format(throttle))
        rospy.logwarn('Brake: {0}'.format(brake))
        rospy.logwarn('Steering: {0}'.format(steering))

        return throttle, brake, steering
