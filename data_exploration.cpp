// C++ Data Exploration
// 4375.004 Intro to Machine Learning
// Chris Talley clt190005

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <math.h>
using namespace std;

double sum(vector<double> nums) {
  double result = 0.0;
  for (vector<double>::iterator iter = nums.begin(); iter != nums.end(); ++iter) {
    result += *iter;
  }
  return result;
}

double mean(vector<double> nums) {
  return sum(nums) / nums.size();
}

double median(vector<double> nums) {
  sort(nums.begin(), nums.end());
  int mid = -1;
  double median = 0.0;
  if (nums.size() % 2 == 0) {
    mid = nums.size() / 2;
    median = (nums[mid] + nums[mid + 1]) / 2.0;
  } else {
    mid = nums.size() / 2;
    median = nums[mid];
  }
  return median;
}

double range(vector<double> nums) {
  sort(nums.begin(), nums.end());
  double range = 0.0;
  range = nums[nums.size() - 1] - nums[0];
  return range;
}

double covar(vector<double> rm, vector<double> medv) {
  int num_points = rm.size();
  double cov = 0.0;
  double summation = 0.0;
  for (int i = 0; i < num_points; i++) {
    summation += (rm[i] - mean(rm)) * (medv[i] - mean(medv));
  }
  cov = summation / (num_points - 1);
  return cov;
}

double sig(vector<double> rm) {
  int num_points = rm.size();
  double var = 0.0;
  double sig = 0.0;
  double summation = 0.0;
  for (int i = 0; i < num_points; i++) {
    summation += pow((rm[i] - mean(rm)), 2);
  }
  var = summation / (num_points);
  sig = sqrt(var);
  
  return sig;
}

double cor(vector<double> rm, vector<double> medv) {
  double cor = 0.0;
  cor = (covar(rm, medv)) / (sig(rm) * sig(medv));
  return cor;
}


void print_stats(vector<double> nums) {
  cout << "Sum:\t" << sum(nums) << endl;
  cout << "Mean:\t" << mean(nums) << endl;
  cout << "Median:\t" << median(nums) << endl;
  cout << "Range:\t" << range(nums) << endl;
  cout << "****************" << endl;
}

int main(int argc, char** argv) {
  ifstream inFS;
  string line;
  string rm_in, medv_in;
  const int MAX_LEN = 1000;
  vector<double> rm(MAX_LEN);
  vector<double> medv(MAX_LEN);

  // Attempt to open file
  cout << "Opening file Boston.csv" << endl;

  inFS.open("Boston.csv");
  if (!inFS.is_open()) {
    cout << "Could not open file Boston.csv" << endl;
    return 1; // 1 = error
  }

  // Using inFS like cin stream
  // Boston.csv should contain two doubles

  cout << "Reading line 1" << endl;
  getline(inFS, line);

  // echo heading
  cout << "heading: " << line << endl;

  int numObservations = 0;
  while (inFS.good()) {
    getline(inFS, rm_in, ',');
    getline(inFS, medv_in, '\n');

    rm.at(numObservations) = stof(rm_in);
    medv.at(numObservations) = stof(medv_in);

    numObservations++;
  }

  rm.resize(numObservations);
  medv.resize(numObservations);

  cout << "new length " << rm.size() << endl;

  cout << "Closing file Boston.csv" << endl;
  inFS.close(); // Done with file

  cout << "Number of records: " << numObservations << endl;

  cout << "\nStats for rm" << endl;
  print_stats(rm);

  cout << "\nStats for medv" << endl;
  print_stats(medv);
  
  cout << "\n Covariance = " << covar(rm, medv) << endl;

  cout << "\n Correlation = " << cor(rm, medv) << endl;
  
  cout << "\nProgram Terminated";

  return 0;
}
