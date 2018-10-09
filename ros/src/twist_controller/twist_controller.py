from pid import PID
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # Use PID controller to predict throttle value
        self.pid_controller = PID(0.1, 0.1, 0.1, 0., 0.5)

        # Use Yaw controller to predict steering angle
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle

        self.last_time = rospy.get_time()

    def control(self, dbw_enabled, current_vel, linear_vel, angular_vel):
        # Reset PID when DBW is disable
        if not dbw_enabled:
            self.pid_controller.reset()
            return 0., 0., 0.

        # Compute current error and sample time for PID
        error = linear_vel - current_vel
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time

        # Retrieve throttle from PID controller
        throttle = self.pid_controller.step(error, sample_time)
        brake = 0

        # Compute braking
        if linear_vel == 0. and current_vel < 0.1:
            # Stop the car!
            brake = 700 # Torque N*m
            throttle = 0

        elif throttle < .01 and error < 0:
            # Brake to decelerate smoothly
            decel = max(throttle, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # Torque N*m
            throttle = 0

        # Retrieve steering from yaw controller
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        self.last_time = current_time

        return throttle, brake, steering
