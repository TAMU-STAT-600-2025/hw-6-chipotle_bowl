// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// Objective only: nll + (lambda/2)*||beta||_F^2
static inline double objective_only(const arma::mat& beta,
                                    const arma::mat& X,
                                    const arma::uvec& y,
                                    const double lambda) {
  
  // Get the number of samples
  const arma::uword n = X.n_rows;
  
  // Scores and softmax computation
  
  arma::mat S = X * beta; // n x K
  arma::vec row_max = arma::max(S, 1); // n
  S.each_col() -= row_max;
  arma::mat eS = arma::exp(S); // n x K
  arma::vec den = arma::sum(eS, 1); // n
  arma::mat P  = eS;
  P.each_col() /= den;
  
  // Negative log-likelihood
  double nll = 0.0;
  for (arma::uword i = 0; i < n; ++i) nll -= std::log(P(i, y(i)));
  
  // Ridge panalization
  const double ridge = 0.5 * lambda * arma::accu(beta % beta);
  return nll + ridge;
}

// Compute term_after_eta D where each column k solves
// (X^T diag(w_k) X + lambda I) d_k = g_k,
// with g = X^T(P - Y_onehot) + lambda*beta.
static inline void compute_term_after_eta(arma::mat& D,
                                          const arma::mat& beta,
                                          const arma::mat& X,
                                          const arma::mat& Xt,
                                          const arma::uvec& y,
                                          const double lambda) {
  
  // Get dimentions
  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  const arma::uword K = beta.n_cols;
  
  // Scores and softmax
  arma::mat S = X * beta; // n x K
  arma::vec row_max = arma::max(S, 1);
  S.each_col() -= row_max;
  arma::mat eS = arma::exp(S);
  arma::vec den = arma::sum(eS, 1);
  arma::mat P  = eS;
  P.each_col() /= den; // n x K
  
  // Gradient: X^T(P - Y_onehot) + lambda*beta
  arma::mat P_for_grad = P;
  for (arma::uword i = 0; i < n; ++i) P_for_grad(i, y(i)) -= 1.0;
  arma::mat G = Xt * P_for_grad + lambda * beta;   // p x K
  
  // Hessian block solve via Cholesky
  for (arma::uword k = 0; k < K; ++k) {
    arma::vec w = P.col(k) % (1.0 - P.col(k)); // n
    
    // Xw = diag(w)*X by row scaling
    arma::mat Xw = X; // n x p
    Xw.each_col() %= w;
    
    arma::mat Hk = Xt * Xw; // p x p
    Hk.diag() += lambda; // + lambda I
    
    arma::mat R;
    arma::vec dk;
    if (arma::chol(R, Hk)) {
      // Solve (R^T) z = g, then R d = z
      arma::vec z = arma::solve(arma::trimatl(R.t()), G.col(k));
      dk = arma::solve(arma::trimatu(R), z);
    } else {
      dk = arma::solve(arma::sympd(Hk), G.col(k));
    }
    D.col(k) = dk;
  }
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

}