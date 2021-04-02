#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

// Tools KalmanFilter::tools_ = Tools();

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;

// remember that we're assuming that there is 0 noise; therefore, u does not need to be specified
}

void KalmanFilter::Predict() {
  
  x_ = F_*x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_*P_*Ft+Q_;

}

void KalmanFilter::Update(const VectorXd &z) {

  MatrixXd I_ = MatrixXd::Identity(4,4);
  
  VectorXd y_ = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S_ = H_ * P_ * Ht + R_;
  MatrixXd Si = S_.inverse();
  MatrixXd K_ =  P_ * Ht * Si;

  x_ = x_ + (K_ * y_);
  P_ = (I_ - K_ * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
  MatrixXd I_ = MatrixXd::Identity(4,4);
  float x = x_(0);
  float y = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float rho = sqrt(x*x+y*y);
  float phi = atan2(y/x);
  float rho_dot = (x*vx+y*vy)/rho;

  VectorXd z_pred = VectorXd(3);
  z_pred << rho,phi,rho_dot;
  
  VectorXd y_ = z - z_pred;

  while (y_(1) < -M_PI){
    y_(1) += 2 * M_PI;
  }
  while (y_(1) > M_PI){
    y_(1) -= 2 * M_PI;
  }

  MatrixXd Ht = H_.transpose();
  MatrixXd S_ = H_ * P_ * Ht + R_;
  MatrixXd Si = S_.inverse();
  MatrixXd K_ =  P_ * Ht * Si;

  x_ = x_ + (K_ * y_);
  P_ = (I_ - K_ * H_) * P_;

}
