// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// 
// Compute 100 * mean(pred != y) for multinomial linear scores.
// 
// X - n x p design matrix
// y - length-n integer labels in {0, 1, ..., K-1}
// beta - p x K coefficient matrix
// 
// Returns error score in [0, 100].
// 
// [[Rcpp::export]]
double error_score_rcpp(const arma::mat& X,
                        const arma::uvec& y,
                        const arma::mat& beta) {
  // Scores matrix n x K
  arma::mat scores = X * beta;
  
  // Get the number of samples
  const arma::uword n = scores.n_rows;
  arma::uvec pred(n);
  
  // For each individual sample determine the predicted class (biggest score)
  for (arma::uword i = 0; i < n; ++i) {
    pred(i) = scores.row(i).index_max(); // in 0..K-1
  }
  
  // Compute accuracy
  arma::uword mismatches = arma::accu(pred != y);
  return 100.0 * static_cast<double>(mismatches) / static_cast<double>(n);
}

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)
// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                               int numIter = 50, double eta = 0.1, double lambda = 1){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int K = max(y) + 1; // number of classes
    int p = X.n_cols;
    int n = X.n_rows;
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    arma::vec objective(numIter + 1); // to store objective values
    
    // Initialize anything else that you may need
    
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
