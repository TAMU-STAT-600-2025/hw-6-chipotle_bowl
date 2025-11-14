#' K-means Clustering Algorithm
#'
#' @param X A numeric matrix of size n x p
#' @param K An integer specifying the number of clusters
#' @param M A K x p matrix of initial cluster centers. If NULL (default), 
#'   K random observations from X will be used as initial centers
#' @param numIter Maximum number of iterations (default is 100)
#'
#' @return A vector of length n containing cluster assignments
#' @export
#'
#' @examples
#' # Give example
MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  n = nrow(X) # number of rows in X
  
  # Check whether M is NULL or not. If NULL, initialize based on K random points from X. If not NULL, check for compatibility with X dimensions.
  
  
  # Call C++ MyKmeans_c function to implement the algorithm
  Y = MyKmeans_c(X, K, M, numIter)
  
  # Return the class assignments
  return(Y)
}