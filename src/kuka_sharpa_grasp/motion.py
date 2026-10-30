"""Bound command derivatives without modifying physical joint velocities."""
import numpy as np


class ArmCommandLimiter:
    def __init__(self, position, velocity_limits, acceleration_limits):
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros_like(self.position)
        self.velocity_limits = np.asarray(velocity_limits, dtype=float)
        self.acceleration_limits = np.asarray(acceleration_limits, dtype=float)
        if (self.position.shape != self.velocity_limits.shape
                or self.position.shape != self.acceleration_limits.shape
                or not np.isfinite(self.position).all()
                or not np.isfinite(self.velocity_limits).all()
                or not np.isfinite(self.acceleration_limits).all()
                or np.any(self.velocity_limits <= 0)
                or np.any(self.acceleration_limits <= 0)):
            raise ValueError('finite positions and positive matching limits are required')

    def update(self, target, dt):
        target = np.asarray(target, dtype=float)
        if target.shape != self.position.shape or not np.isfinite(target).all() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('a finite matching target and positive timestep are required')
        error = target - self.position
        acceleration_step = self.acceleration_limits * dt
        # Reserve one integration step of braking distance before the goal.
        stopping_speed = np.sqrt(acceleration_step**2 + 2. * self.acceleration_limits * np.abs(error)) - acceleration_step
        desired = np.sign(error) * np.minimum(
            np.minimum(40. * np.abs(error), stopping_speed), self.velocity_limits)
        delta = np.clip(desired - self.velocity, -self.acceleration_limits * dt,
                        self.acceleration_limits * dt)
        self.velocity += delta
        self.position += self.velocity * dt
        return self.position.copy()
