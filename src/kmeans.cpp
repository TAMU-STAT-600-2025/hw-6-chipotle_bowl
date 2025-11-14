// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                            const arma::mat& M, int numIter = 100){
    // All input is assumed to be correct
    // Inputs:
    // X n x p matrix for input data
    // K fixed number of clusters
    // M initial centers for all the clusters of data
    // numIter max number of iterations allowed for the algorithm
    
    // Initialize some parameters
    int n = X.n_rows;
    int p = X.n_cols;
    arma::uvec Y(n); // to store cluster assignments
    
    // Initialize any additional parameters if needed
    arma::vec X_sq = arma::sum(arma::square(X), 1);
    arma::mat M_current = M; // Current centers
    arma::uvec cluster_index; // Vector for the index of all the centers 
    
    // For loop with kmeans algorithm
    for (int iter = 1; iter <= numIter; iter++) {
      // ||X_i - mu_k||^2 = (X_i - mu_k)^T(X_i - mu_k) = ||X_i||^2 + ||mu_k||^2 - 2 * X_i^T * mu_k
      arma::vec M_sq = arma::sum(arma::square(M_current), 1);
      arma::mat XM = X * M_current.t();
      
      // Pair-wise distance between the points and the cluster centers || \mu_i - x_i ||
      arma::mat sq_euc_dist = arma::repmat(X_sq, 1, M_sq.size()) + 
        arma::repmat(M_sq.t(), X_sq.size(), 1) - 2 * XM;
      
      // Re-assign the correspondent new cluster for each point
      cluster_index = arma::index_min(sq_euc_dist, 1);
      
      // Check if a cluster has disappeared
      arma::uvec unique_clusters = arma::unique(cluster_index);
      
      // As a sanity check stop if any of the clusters disappeared
      if (unique_clusters.n_elem != K) {
        Rcpp::stop("One of the clusters has disappeared");
      }
      
      // Assign new M
      arma::mat M_new = M_current;
      
      // Re-compute the centers of the clusters
      for (int k = 0; k < K; k++) {
        M_new.row(k) = arma::mean(X.rows(arma::find(cluster_index == k)), 0);
      }
      
      // Check of the mean of the clusters did not change between iterations
      if (arma::approx_equal(M_current, M_new, "absdiff", 1e-10)) {
        Rcpp::Rcout << "Centroids have converged" << std::endl;
        break;
      }
      
      // Update the new centers.
      M_current = M_new;
      
      // Stop if the max number of iterations has been reached
      if (iter == numIter) {
        Rcpp::Rcout << "Maximum number of iterations reached" << std::endl;
      }
    }
    
    // Returns the vector of cluster assignments
    Y = cluster_index + 1;
    return(Y);
}

