#' Title
#'
#' @param X train data
#' @param y train labels
#' @param numIter number of iterations
#' @param eta eta factor
#' @param lambda regularization factor
#' @param beta_init initial beta value
#'
#' @return return a list with the beta values and the array of objective values
#' @export
#'
#' @examples
#' # Give example
#' X <- matrix(rbind(c(1,-1,1), c(1,-1,1.25), c(1,1,2)))
#' y <- c(0,0,1)
#' beta <- NULL
#' out <- LRMultiClass(X, y, beta_init = NULL, numIter = 50, eta = 0.1, lambda = 1)

LRMultiClass <- function(X, y, beta_init = NULL, numIter = 50, eta = 0.1, lambda = 1){
  
  # Compatibility checks from HW3 and initialization of beta_init
  
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if (!all(X[, 1] == 1)) {
    stop('First columns must be all ones.')
  }
  
  # Check for compatibility of dimensions between X and Y
  n <- nrow(X)
  if (n != length(y)) {
    stop('X and y (training data) have different number of samples.')
  }

  ## Check eta is positive
  if (!is.numeric(eta) ||
      length(eta) != 1 || !is.finite(eta) || eta <= 0) {
    stop('eta must be a positive and finite numeric value.')
  }
  
  ## Check lambda is non-negative
  if (!is.numeric(lambda) ||
      length(lambda) != 1 || !is.finite(lambda) || lambda < 0) {
    stop('lambda must be a non-negative and finite numeric value.')
  }
  
}