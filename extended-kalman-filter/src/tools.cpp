#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if(estimations.size() != ground_truth.size() || estimations.size() == 0){
    return rmse;
  }
  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];
    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }
  //calculate the mean
  rmse = rmse/estimations.size();
  //calculate the squared root
  rmse = rmse.array().sqrt();
  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  //check division by zero
  if(fabs(c1) < 0.0001){
    return Hj;
  }
  //check division by zero
  if(fabs(c3) < 0.0001){
    c3 = 0.0001;
  }

  //compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

return Hj;
}

VectorXd Tools::ToCartesian(const VectorXd& polar) {
    assert(polar.rows() == 3 && polar.cols() == 1);

    VectorXd x = VectorXd(4);

    float ro = polar(0);
    float phi = polar(1);
    float ro_dot = polar(2);

    float px = ro*cos(phi);
    float py = ro*sin(phi);
    float vx = ro_dot*cos(phi);
    float vy = ro_dot*sin(phi);

    x << px, py, vx, vy;
    return x;
}

float Tools::Normalize(float angle) {
    return atan2(sin(angle), cos(angle));
}

