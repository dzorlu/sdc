# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

# Reflections
Finding the right coefficients took few iterations and examination of different models to provide a robust enough solution. The implementation of a basic PID controller is just few lines of code, but the tuning is harder.

PID is made up of three components: P stands for proportional and it corrects the steering angle of the vehicle in proportion to the cross-track error. The I is the cumulative sums of all CTEs up to that point. Integral component is meant to eliminate any systemic errors that accumulate over time. The last component, D, is the derivate of CTE. The D component is used to compliment the P component for a quicker convergence.

I tried several approaches to optimize the controller including SGD and Twiddle but ended up using hand-tuned weights. It turned out that the hand-tuned weights provide the most robust and consistent solution in the toy problem. I think the reason why the Twiddle didn't work was because the I part was driving the system as the car was swerving with large angles. This might have happened because there is really no systematic bias in the simulated environment. Hence I ended up setting the I to a very small value at the end.

The parameters are set as follows:

```
Kp = 0.1;
Ki = 0.0001;
Kd = 4.0;
```
