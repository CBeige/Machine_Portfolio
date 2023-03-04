#ifndef LogReg_h
#define LogReg_h

#include "Eigen/Dense"
#include <list>

class LogReg {
public:
  LogReg() {}
  Eigen::MatrixXd Sigmoid(Eigen::MatrixXd Z);

  std::tuple<Eigen::MatrixXd, double, double>
  Propogate(Eigen::MatrixXd W, double b, Eigen::MatrixXd X, Eigen::MatrixXd y,
            double lambda);
  std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double,
             std::list<double>>
  Optimize(Eigen::MatrixXd W, double b, Eigen::MatrixXd X, Eigen::MatrixXd y,
           int num_iter, double learning_rate, double lambda, bool log_cost);
  Eigen::MatrixXd Predict(Eigen::MatrixXd W, double b, Eigen::MatrixXd X);
};

#endif
