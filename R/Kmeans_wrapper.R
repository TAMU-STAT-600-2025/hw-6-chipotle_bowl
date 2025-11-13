#' Title
#'
#' @param X 
#' @param K 
#' @param M 
#' @param numIter 
#'
#' @return Explain return
#' @export
#'
#' @examples
#' # Generate simple 2D data with 3 clusters
#' set.seed(123)
#' n <- 150
#' p <- 2
#' K <- 3
#' 
#' # Create three clusters
#' cluster1 <- matrix(rnorm(50*p, mean = 0), ncol = p)
#' cluster2 <- matrix(rnorm(50*p, mean = 5), ncol = p)
#' cluster3 <- matrix(rnorm(50*p, mean = c(2.5, 5)), ncol = p)
#' X <- rbind(cluster1, cluster2, cluster3)
#' 
#' # Run K-means clustering
#' result <- MyKmeans(X, K = 3)
#' 
#' # Check cluster assignments
#' table(result)
#' 
#' # Example with custom initial centers
#' initial_centers <- X[c(1, 51, 101), ]
#' result2 <- MyKmeans(X, K = 3, M = initial_centers)
#' 
#' # Check cluster assignments
#' table(result2)

MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  n = nrow(X) # number of rows in X
  
  # Check whether M is NULL or not. If NULL, initialize based on K random points from X. If not NULL, check for compatibility with X dimensions.
  if (is.null(M)) {
    if (K > nrow(X)) stop("K cannot exceed number of rows in X")
    random_row_index <- sample(1:nrow(X), K, replace = FALSE)
    M <- X[random_row_index, ]
  } else if (!is.matrix(M) || nrow(M) != K || ncol(M) != ncol(X)) {
    stop(paste("Initial centroid should be a K by p matrix"))
  }
  
  # Call C++ MyKmeans_c function to implement the algorithm
  Y = MyKmeans_c(X, K, M, numIter)
  
  # Return the class assignments
  return(Y)
}