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
  
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void KalmanFilter::Update(const VectorXd &z) {
  
  VectorXd y_ = z - H_ * x_;

  while (y_(1) < -M_PI){
    y_(1) += 2 * M_PI;
  }
  while (y_(1) > M_PI){
    y_(1) -= 2 * M_PI;
  }

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;

  x_ = x_ + (K * y_);

  int x_shape = x_.size();
  MatrixXd I = MatrixXd::Identity(x_shape,x_shape);

  P_ = (I - K * H_) * P_;

}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float rho = sqrt(px*px+py*py);
  float phi = atan2(py/px);
  float rho_dot = (px*vx+py*vy)/rho;

  VectorXd z_pred(3);
  z_pred << rho,phi,rho_dot;
  
  VectorXd y = z - z_pred;

  while (y(1) < -M_PI){
      y(1) += 2 * M_PI;
  }
  while (y(1) > M_PI){
    y(1) -= 2 * M_PI;
  }

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;

  x_ = x_ + (K * y);

  int x_shape = x_.size();
  MatrixXd I = MatrixXd::Identity(4,4);

  P_ = (I - K * H_) * P_;

}
