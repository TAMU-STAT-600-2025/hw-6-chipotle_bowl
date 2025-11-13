test_that("multiplication works", {
  
  # Load data
  path <- '/Users/leonardocazares/Desktop/Fall 2025/Reproducible_Computations/HWs/hw-6-chipotle_bowl/data/mydata.rda'
  load(path)
  mydata <- cbind(Intercept = 1, mydata)

  # 2. Extract X = first three columns (as matrix)
  X <- as.matrix(mydata[, 1:3])
  
  # 3. Extract y = fourth column (as vector)
  y <- mydata[, 4]
  
  # 4. Apply the previos R based LR MultiClass to the actual data
  output_R_previous <- LRMultiClass_R(X,
                 y,
                 numIter = 50,
                 eta = 0.1,
                 lambda = 1,
                 beta_init = NULL)
  
  # 4. Apply the new Rcpp based LR MultiClass to the actual data
  output_R <- LRMultiClass(X,
               y,
               beta_init = NULL,
               numIter = 50,
               eta = 0.1,
               lambda = 1)
  
  expect_equal(dim(output_R_previous$beta), dim(output_R$beta))
  expect_equal(output_R_previous$beta, output_R$beta)

})
