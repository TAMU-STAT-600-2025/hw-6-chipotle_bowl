error_score <- function(X, y, beta) {
  scores <- X %*% beta # Compute scores
  pred <- max.col(scores, ties.method = "first") - 1 # Take the class with the maximum score and put classes from 0 to K-1
  return(100 * mean(pred != y)) # Return error score
}

obj_grad_newton <- function(beta, X, y, lambda = 1) {
  n <- nrow(X) # Number of samples
  p <- ncol(X) # Number of features
  K <- ncol(beta) # Number of samples, number of features and number of classes
  
  #1) Get the matrix of inner products S
  S  <- X %*% beta # Computes matrix of probabilities pk (nxK).
  
  ##2) Get the matrix of probabilities pk
  eS <- exp(S) # Exponential of pk matrix
  den <- rowSums(eS) # Sum of exponentials for each row
  P  <- eS / den # Matrix of probabilities
  
  ##3) Get the objective function
  idx <- cbind(seq_len(n), y + 1) # For each sample take the real class
  nll <- -sum(log(P[idx])) # Sum the -log-prob. for each sample
  obj <- nll + (lambda / 2) * sum(beta * beta) # Compute the final objective function
  
  ##4) Compute the gradients G
  P_for_grad <- P # Matrix of gradients, initialized as P
  P_for_grad[idx] <- P_for_grad[idx] - 1 # Subtract one in each row's true class
  G <- crossprod(X, P_for_grad) + lambda * beta # p x K
  
  ##5) Term after the eta, i.e. (Hessian)^{-1} * Gradient
  D <- matrix(0.0, nrow = p, ncol = K)
  for (k in seq_len(K)) {
    w  <- P[, k] * (1 - P[, k])        # Vector of diagonal entries of W_k
    Hk <- crossprod(X, X * w)          # X^T W_k X using row-wise weighting
    diag(Hk) <- diag(Hk) + lambda      # Addition reg. coefficient
    
    # Inverting matrix
    R  <- chol(Hk)
    yk <- forwardsolve(t(R), G[, k], upper.tri = FALSE, transpose = FALSE)
    D[, k] <- backsolve(R, yk, upper.tri = TRUE)
  }
  
  list(
    objective = obj,
    gradient = G,
    term_after_eta = D,
    probs = P
  )
}

# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)


LRMultiClass <- function(X,
                         y,
                         Xt,
                         yt,
                         numIter = 50,
                         eta = 0.1,
                         lambda = 1,
                         beta_init = NULL) {
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  
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
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  
  classes <- sort(unique(y)) # Sort classes
  if (any(classes != 0:(length(classes) - 1))) {
    stop("Classes are not labeled from 0 to K-1.")
  }
  K <- length(classes) # Number of classes
  p <- ncol(X) # Number of features
  if (is.null(beta_init)) {
    beta <- matrix(0, nrow = p, ncol = K) # Initializa beta with zeros
  }
  else {
    if (!is.matrix(beta_init) || any(dim(beta_init) != c(p, K)))
      stop("beta_init must be a pxK matrix.")
    beta <- beta_init # Initialize beta with the given beta_init
  }
  
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  objective   <- numeric(numIter + 1)

  og0 <- obj_grad_newton(beta, X, y, lambda = lambda)
  objective[1] <- og0$objective
  
  ##list(objective = obj, gradient = G, term_after_eta = D, probs = P)
  #
  ### Newton's method cycle - implement the update EXACTLY numIter iterations
  ###########################################################################
  for (t in seq_len(numIter)) {
    # Compute gradient and "term after eta" at current beta
    og <- obj_grad_newton(beta, X, y, lambda = lambda)
    
    # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
    beta <- beta - eta * og$term_after_eta
    
    # Objective at the updated beta
    objective[t + 1] <- obj_grad_newton(beta, X, y, lambda = lambda)$objective

  }
  
  
  # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
  
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(
    list(
      beta = beta,
      objective =  objective
    )
  )
}