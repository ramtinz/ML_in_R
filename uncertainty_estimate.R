# In this code, we introduce a new function called noise_predict that adds noise to the data and predicts probabilities using the logistic regression model. We then use the boot function from the boot library to perform bootstrapping on the test data and estimate confidence intervals for the predicted probabilities. Finally, we extract the confidence intervals from the bootstrapping results and add them to the original probabilities. The confidence interval for each individual represents the estimated uncertainty level for the predicted probability.

# Load required libraries
library(caret)
library(glmnet)
library(boot)

# Load data
data(default)

# Split data into training and test sets
set.seed(123)
train_ind <- createDataPartition(default$default, p = 0.8, list = FALSE)
train <- default[train_ind, ]
test <- default[-train_ind, ]

# Fit model
model <- train(default ~ ., data = train, method = "glmnet")

# Function to add noise to data and predict probabilities
noise_predict <- function(data, noise_level) {
  # Add noise to data
  noisy_data <- data + rnorm(nrow(data), mean = 0, sd = noise_level)
  
  # Predict probabilities
  probs <- predict(model, newdata = noisy_data, type = "prob")
  
  return(probs)
}

# Bootstrapping function to estimate confidence intervals
bootstrapped_probs <- function(data, indices) {
  # Select data for bootstrapped sample
  bootstrapped_data <- data[indices, ]
  
  # Add noise and predict probabilities
  probs <- noise_predict(bootstrapped_data, noise_level = 0.5)
  
  return(probs)
}

# Perform bootstrapping to estimate confidence intervals
boot_results <- boot(data = test, statistic = bootstrapped_probs, R = 1000)

# Extract confidence intervals from bootstrapping results
ci <- boot.ci(boot_results)

# Add confidence intervals to original probabilities
probs <- noise_predict(test, noise_level = 0)
probs$ci_lower <- ci$lower[,1]
probs$ci_upper <- ci$upper[,1]
