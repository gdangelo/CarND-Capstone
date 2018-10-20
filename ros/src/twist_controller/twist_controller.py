from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # Use PID controller to predict throttle value
        self.pid_controller = PID(0.1, 0.1, 0.1, decel_limit, accel_limit)

        # Use Yaw controller to predict steering angle
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        # Low pass filter for velocity
        tau = 0.5  # cutoff freq
        ts  = 0.02  # sample time
        self.vel_lowpass = LowPassFilter(tau, ts)
        
        # Vehicles properties
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
    
    def reset(self):
        self.pid_controller.reset()
        
    def control(self, dbw_enabled, current_vel, linear_vel, angular_vel):
        # Reset PID when DBW is disable
        if not dbw_enabled:
            self.reset()
            return 0., 0., 0.
        
        #current_vel = self.vel_lowpass.filt(current_vel)

        # Compute current error and sample time for PID
        error = linear_vel - current_vel
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time

        # Retrieve throttle from PID controller
        throttle = self.pid_controller.step(error, sample_time)
        
        if throttle > 0:
            brake = 0.0
        else:
            decel = -throttle
            if decel < self.brake_deadband:
                decel = 0.0
            brake = 2*decel * (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY) * self.wheel_radius # Torque N*m
            throttle = 0.0

        # Retrieve steering from yaw controller
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        self.last_time = current_time

        return throttle, brake, steering
