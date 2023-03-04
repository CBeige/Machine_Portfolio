#include "NaiveBayes.h"
#include <cmath>
#include <unordered_set>

#define M_PI 3.14159265358979323846 // pi

void stats(Matrix &m, int col_num, double &mean, double &var) {
  double sum = 0, sq_sum = 0;
  for (int i = 0; i < m.getRows(); i++) {
    sum += m[i][0];
    sq_sum += pow(m[i][0], 2);
  }
  mean = sum / m.getRows();
  var = sq_sum / m.getRows() - pow(mean, 2);
  if (var < 0) {
    var = 0;
  }
}

Matrix unique(const Matrix &m) {
  unordered_set<double> set;
  for (int i = 0; i < m.getRows(); i++) {
    if (set.find(m[i][0]) == set.end()) {
      set.insert(m[i][0]);
    }
  }
  Matrix res(set.size(), 1);
  int index = 0;
  for (auto x : set) {
    res[index++][0] = x;
  }
  return res;
}

Matrix extract_class(Matrix &X, double c, Matrix &y) {
  vector<vector<double>> res;
  for (int i = 0; i < y.getRows(); i++) {
    if (y[i][0] == c) {
      vector<double> row(X.getCols());
      for (int j = 0; j < X.getCols(); j++) {
        row[j] = X[i][j];
      }
      res.push_back(row);
    }
  }
  Matrix result(res);
  return result;
}

void NaiveBayes::fit(Matrix &train_X, Matrix &train_y) {
  X = train_X;
  y = train_y;
  classes = unique(train_y);
  parameters = vector<vector<pair<double, double>>>(classes.getRows());
  for (int i = 0; i < classes.getRows(); i++) {
    double &c = classes[i][0];
    Matrix x_c = extract_class(X, c, y);
    for (int col = 0; col < x_c.getCols(); col++) {
      double mean, var;
      stats(x_c, col, mean, var);
      parameters[i].push_back(make_pair(mean, var));
    }
  }
}

double NaiveBayes::calculate_likelihood(double mean, double var, double x) {
  double eps = 1e-4;
  double coeff = 1.0 / sqrt(2 * M_PI * var + eps);
  double expon = exp(-(pow(x - mean, 2) / (2 * var + eps)));
  return coeff * expon;
}

double NaiveBayes::calculate_prior(double c) {
  double res = 0;
  int size = y.getRows();
  for (int i = 0; i < size; i++) {
    if (y[i][0] == c) {
      res++;
    }
  }
  return res / size;
}

int argmax(vector<double> &vec) {
  double max = -1e6;
  int index = -1;
  for (int i = 0; i < vec.size(); i++) {
    if (max < vec[i]) {
      max = vec[i];
      index = i;
    }
  }
  return index;
}

double NaiveBayes::classify(Matrix &x) {
  vector<double> posteriors(classes.getRows());
  for (int i = 0; i < classes.getRows(); i++) {
    double &c = classes[i][0];
    double posterior = calculate_prior(c);
    for (int j = 0; j < x.getCols(); j++) {
      double &feature_val = x[0][j];
      pair<double, double> &params = parameters[i][j];
      double likelihood =
          calculate_likelihood(params.first, params.second, feature_val);
      posterior *= likelihood;
    }
    posteriors[i] = posterior;
  }
  int index = argmax(posteriors);
  return classes[index][0];
}

Matrix NaiveBayes::predict(Matrix &train_X) {

  Matrix res(train_X.getRows(), 1);
  for (int i = 0; i < res.getRows(); i++) {
    double y_pred =
        classify(Matrix(train_X.return_row(i), train_X.getCols(), "row"));
    res[i][0] = y_pred;
  }
  return res;
}
