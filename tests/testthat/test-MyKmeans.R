test_that("multiplication works", {
  
  # Simulate a good case
  X <- rbind(
    matrix(rnorm(20, 0), ncol = 4),
    matrix(rnorm(20, 50), ncol = 4),
    matrix(rnorm(20, 100), ncol = 4)
  )
  
  good_M <- X[c(1,6,11),]
  
  K <- 3
  
  output_R_previous <- MyKmeans_R(X, K, good_M, numIter = 100)
  
  output_R <- MyKmeans(X, K, good_M, numIter = 100)
  
  confusion_matrix <- table(output_R_previous, output_R)
  
  # Check that each row has exactly one non-zero entry
  row_counts <- apply(confusion_matrix != 0, 1, sum)
  expect_true(all(row_counts == 1))
  
  # Check that each column has exactly one non-zero entry
  col_counts <- apply(confusion_matrix != 0, 2, sum)
  expect_true(all(col_counts == 1))
})
