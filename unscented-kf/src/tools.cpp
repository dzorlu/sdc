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
    float v  = sqrt(vx * vx + vy * vy);

    x << px, py, v, 0, 0;
    return x;
}
