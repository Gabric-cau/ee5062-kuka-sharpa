"""Release permission from a declared simulated instrumented-table sensor."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SupportConfig:
    minimum_load_n: float
    confirmation_seconds: float = .12
    filter_seconds: float = .03
    maximum_sample_gap_seconds: float = .02
    descent_step_m: float = .002
    maximum_extra_descent_m: float = .012
    descent_seconds: float = .4

    def __post_init__(self):
        if not all(math.isfinite(value) and value > 0 for value in self.__dict__.values()):
            raise ValueError('support settings must be finite and positive')
        if self.descent_step_m > self.maximum_extra_descent_m:
            raise ValueError('descent step exceeds travel limit')


class TableSupportGate:
    def __init__(self, config):
        self.config = config
        self.last_time = None
        self.filtered_load_n = 0.
        self.support_since = None
        self.confirmed = False

    def update(self, timestamp, load_n):
        if not math.isfinite(timestamp) or not math.isfinite(load_n) or load_n < 0:
            raise ValueError('finite time and nonnegative measured load are required')
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError('support timestamps must strictly increase')
        dt = 0. if self.last_time is None else timestamp - self.last_time
        if self.last_time is None or dt > self.config.maximum_sample_gap_seconds:
            self.filtered_load_n = 0.
            self.support_since = None
        else:
            self.filtered_load_n += -math.expm1(-dt / self.config.filter_seconds) * (load_n - self.filtered_load_n)
        supported = load_n >= self.config.minimum_load_n and self.filtered_load_n >= self.config.minimum_load_n
        if not supported:
            self.support_since = None
        elif self.support_since is None:
            self.support_since = timestamp
        duration = 0. if self.support_since is None else timestamp - self.support_since
        self.confirmed = supported and duration + 1e-12 >= self.config.confirmation_seconds
        self.last_time = timestamp
        return dict(raw_load_n=load_n, filtered_load_n=self.filtered_load_n,
                    continuous_support_seconds=duration, confirmed=self.confirmed)

    def fresh_confirmation(self, timestamp):
        return (self.last_time is not None and self.confirmed
                and 0 <= timestamp - self.last_time <= self.config.maximum_sample_gap_seconds)
