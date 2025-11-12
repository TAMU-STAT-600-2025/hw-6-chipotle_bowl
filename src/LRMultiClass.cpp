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

//
//Multinomial logistic: objective, gradient, and per-class (H^{-1} * grad) term.
//beta  : p x K coefficient matrix
//X     : n x p design matrix
//y     : length-n integer labels in {0, 1, ..., K-1}
//lambda: L2 regularization strength (ridge)
//Returns a list:
// - objective      : scalar (negative log-likelihood + ridge)
// - gradient       : p x K matrix (X^T (P - Y_onehot) + lambda * beta)
// - term_after_eta : p x K matrix with columns H_k^{-1} * gradient_k,
//                    where H_k = X^T diag(p_k (1 - p_k)) X + lambda I_p
// - probs          : n x K matrix of class probabilities
//
// [[Rcpp::export]]
Rcpp::List obj_grad_newton_rcpp(const arma::mat& beta,
                                const arma::mat& X,
                                const arma::uvec& y,
                                const double lambda = 1.0) {
  
  // 1) Get the dimentions for all the matrices
  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  const arma::uword K = beta.n_cols;

  // 2) Softmax probabilities P
  arma::vec row_max = arma::max(S, 1); // n x 1
  S.each_col() -= row_max; // stabilize
  arma::mat eS = arma::exp(S); // element-wise exp
  arma::vec den = arma::sum(eS, 1); // n x 1 (row sums)
  arma::mat P  = eS; // n x K
  P.each_col() /= den; // divide each row by its sum

  // 3) Objective function NLL + (lambda/2) * ||beta||_F^2
  double nll = 0.0; // Initial value of NLL
  for (arma::uword i = 0; i < n; ++i) {
    const arma::uword yi = y(i); // 0..K-1
    // Update NLL
    nll -= std::log(P(i, yi));
  }
  const double ridge = 0.5 * lambda * arma::accu(beta % beta); // Add regularization
  const double obj = nll + ridge; // Complete objective value
  
  
  // 4) Gradient X^T (P - Y_onehot) + lambda * beta
  // Implement P_for_grad = P; P[i, y_i] -= 1
  arma::mat P_for_grad = P; // n x K
  for (arma::uword i = 0; i < n; ++i) {
    P_for_grad(i, y(i)) -= 1.0;
  }
  arma::mat G = X.t() * P_for_grad + lambda * beta; // p x K
  


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
