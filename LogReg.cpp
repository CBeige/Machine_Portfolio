// C++ Log Reg
// 4375.004 Intro to Machine Learning
// Chris Talley clt190005
#include "LogReg.h"
#include "ETL.h"
#include "Eigen/Dense"

#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>

using namespace std;

Eigen::MatrixXd LogReg::Sigmoid(Eigen::MatrixXd Z) {
  return 1 / (1 + (-Z.array()).exp());
}

tuple<Eigen::MatrixXd, double, double>
LogReg::Propogate(Eigen::MatrixXd W, double b, Eigen::MatrixXd X,
                  Eigen::MatrixXd y, double lambda) {
  int m = y.rows();

  Eigen::MatrixXd Z = (W.transpose() * X.transpose()).array() + b;
  Eigen::MatrixXd A = Sigmoid(Z);

  auto cross_entropy =
      -(y.transpose() * (Eigen::VectorXd)A.array().log().transpose() +
        ((Eigen::VectorXd)(1 - y.array())).transpose() *
            (Eigen::VectorXd)(1 - A.array()).log().transpose()) /
      m;

  double l2_reg_cost = W.array().pow(2).sum() * (lambda / (2 * m));

  double cost =
      static_cast<const double>((cross_entropy.array()[0])) + l2_reg_cost;

  Eigen::MatrixXd dw =
      (Eigen::MatrixXd)(((Eigen::MatrixXd)(A - y.transpose()) * X) / m) +
      ((Eigen::MatrixXd)(lambda / m * W)).transpose();

  double db = (A - y.transpose()).array().sum() / m;

  return make_tuple(dw, db, cost);
}

tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, std::list<double>>
LogReg::Optimize(Eigen::MatrixXd W, double b, Eigen::MatrixXd X,
                 Eigen::MatrixXd y, int num_iter, double learning_rate,
                 double lambda, bool log_cost) {
  list<double> costsList;

  Eigen::MatrixXd dw;
  double db, cost;

  for (int i = 0; i < num_iter; i++) {
    tuple<Eigen::MatrixXd, double, double> propogate =
        Propogate(W, b, X, y, lambda);
    tie(dw, db, cost) = propogate;

    W = W - (learning_rate * dw).transpose();
    b = b - (learning_rate * db);

    if (i % 100 == 0) {
      costsList.push_back(cost);
    }

    if (log_cost && i % 100 == 0) {
      cout << "cost after iter " << i << ": " << cost << endl;
    }
  }
  return make_tuple(W, b, dw, db, costsList);
}

Eigen::MatrixXd LogReg::Predict(Eigen::MatrixXd W, double b,
                                Eigen::MatrixXd X) {
  int m = X.rows();

  Eigen::MatrixXd y_pred = Eigen::VectorXd::Zero(m).transpose();

  Eigen::MatrixXd Z = (W.transpose() * X.transpose()).array() + b;
  Eigen::MatrixXd A = Sigmoid(Z);

  for (int i = 0; i < A.cols(); i++) {
    if (A(0, i) <= 0.5) {
      y_pred(0, i) = 0;
    } else {
      y_pred(0, i) = 1;
    }
  }
  return y_pred.transpose();
}

int main(int argc, char *argv[]) {
  ETL etl(argv[1], argv[2], argv[3]);

  vector<vector<string>> dataset = etl.readCSV();

  int rows = dataset.size();
  int cols = dataset[0].size();

  cout << "Reading Dataset" << endl;
  Eigen::MatrixXd dataMatrix = etl.CSVtoEigen(dataset, rows, cols);

  Eigen::MatrixXd norm = etl.Normalize(dataMatrix, false);

  cout << "Splitting Dataset" << endl;
  Eigen::MatrixXd X_train, y_train, X_test, y_test;
  tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
      split_data = etl.TrainTestSplit(norm, 0.8);
  tie(X_train, y_train, X_test, y_test) = split_data;

  auto start = chrono::steady_clock::now();

  cout << "Creating LR model from Dataset" << endl;
  LogReg lrm;

  int dims = X_train.cols();
  Eigen::MatrixXd W = Eigen::VectorXd::Zero(dims);
  double b = 0.0;
  double lambda = 0.0;
  bool log_cost = false;
  double learning_rate = 0.0;
  int num_iter = 10000;

  Eigen::MatrixXd dw;
  double db;
  list<double> costs;
  tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, list<double>>
      optimize = lrm.Optimize(W, b, X_train, y_train, num_iter, learning_rate,
                              lambda, log_cost);
  tie(W, b, dw, db, costs) = optimize;

  auto end = chrono::steady_clock::now();

  cout << "Optimized Coefficients:" << endl;
  cout << "W: " << dw << endl;
  cout << "B: " << db << endl;
  Eigen::MatrixXd y_pred_test = lrm.Predict(W, b, X_test);
  Eigen::MatrixXd y_pred_train = lrm.Predict(W, b, X_train);

  auto train_acc = (100 - (y_pred_train - y_train).cwiseAbs().mean() * 100);
  auto test_acc = (100 - (y_pred_test - y_test).cwiseAbs().mean() * 100);

  cout << "Train Accuracy: " << train_acc << endl;
  cout << "Test Accuracy: " << test_acc << endl;

  cout << "Elapsed Time in milliseconds: "
       << chrono::duration_cast<chrono::milliseconds>(end - start).count()
       << " ms" << endl;
  return 0;
}
