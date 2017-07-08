

### project description
The goal of the project is to provide actuator commands to the simulator that successfully steers the vehicle along the track. The immediate estimated track path -waypoints for the coming stretch of track.- is provided through the websocket in the simulator along with the telemetry (position, velocity, heading).

### cost function and constraints
Cost function is defined in terms of CTE and PSI errors, in addition to the magnitude of actuators as well as the delta between sequential actuations to ensure smooth transitions between states. Secondly, we define the vehicle model that defines the state transitions. The model is defined in terms of constraints. Both the cost function and the constraints are defined in the `MPC.operator` method.

## MPC process
The algorithm fits a third degree polynomial to the incoming waypoints, which in turn is fed into the optimizer in `MPC.solver` to determine the steering and throttle variables that minimize the cost function over time steps. Number of time steps and the interval are hyper-parameters to the model and present a tradeoff. At each step, the process is repeated. Only the actuator parameters from the first time steps are kept and rest are discarded.

## Simulating latency
