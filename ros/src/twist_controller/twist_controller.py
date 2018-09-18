from pid import PID
from yaw_controller import YawController

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # Use PID controller to predict throttle value
        self.pid_controller = PID(0.1, 0.1, 0.1, 0., 1.)

        # Use Yaw controller to predict steering angle
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

    def control(self, dbw_enabled):
        # Reset PID when DBW is disable
        if not dbw_enabled:
            self.pid_controller.reset()
            return None

        # Compute current error for PID
        # TODO

        # Return throttle, brake, steer
        return 1., 0., 0.
