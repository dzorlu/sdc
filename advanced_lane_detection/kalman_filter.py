import numpy as np

class KalmanFilter1D(object):
  def __init__(self, initial_state):
    """State is a measurement_dimension-al array"""
    # These are updates at every prediction and update steps.
    self._state = initial_state
    self._noise = 1
    # Measurement noise is an array decreasing monotonically
    # Pixels in the horizon adjust faster
    self.measurement_noise = self.noise * np.linspace(10, 25, 721)

  def update(self,update):
    # noise gets smaller
    self.noise = (self.noise  * self.measurement_noise) / (self.noise + self.measurement_noise)
    # update the state per noise proportions.
    # kalman_gain x _residual gives you the adjustment in pixels
    _kalman_gain = self.noise / (self.noise + self.measurement_noise)
    _residual = update - self.state
    self.state = _kalman_gain * _residual + self.state

  def predict(self,):
    # random prediction.
    self.noise = self.noise + self.measurement_noise
    _random_error = np.sqrt(self.noise) * np.random.rand(1)
    self.s = self.s + _random_error

  def step(self, update):
    self.update(update)
    self.predict()

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self,values):
    self._state = values

  @property
  def noise(self):
    return self._noise

  @noise.setter
  def noise(self, value):
    self._noise = value

#
# m, s = 25,1
# i = 0
# _s = []
# while i < 100:
#     s = (m * s) / (m + s)
#     s +=  m
#     _s.append(s)
#     i += 1
