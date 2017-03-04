import numpy as np

class KalmanFilter1D(object):
  def __init__(self, initial_state, measurement_dimension=720, pixels_second_adjustment=50):
    # These are updates at every prediction and update steps.
    self.state_noise = 1
    self.s = initial_state
    # Measurement noise is an array decreasing monotonically
    # Pixels in the horizon adjust twice faster
    _adj = np.linspace(pixels_second_adjustment , pixels_second_adjustment//2, measurement_dimension)

  def update(self,update):
    # noise gets smaller
    self.state_noise = (self.state_noise  * self.measurement_noise) / (self.state_noise + self.measurement_noise)
    # update the state per noise proportions.
    # kalman_gain x _residual gives you the adjustment in pixels
    _kalman_gain = self.state_noise / (self.state_noise + self.measurement_noise)
    _residual = update - self.s
    self.s = _kalman_gain * _residual + self.s

  def predict(self,):
    # random prediction.
    self.state_noise = self.state_noise + self.measurement_noise
    _random_error = np.sqrt(self.state_noise) * np.random.rand(1)
    self.s = self.s + _random_error

  def step(self, update):
    self.update(update)
    self.predict()

  @property
  def state(self):
    return self.s

  @property
  def noise(self):
    return self.state_noise
