# noise simulation and estimate the distribution of noise from the data

# In this code, we use the fitdistr function from the MASS library to estimate the distribution of noise in the data. We then use the noise_predict function to add noise to the test set data and predict probabilities using the ensemble model. The noise level is determined by sampling from the estimated noise distribution. Finally, we use the apply function with the sd function to calculate the standard deviation of the predicted probabilities for each instance, which serves as a measure of the uncertainty level.

# Load required libraries
library(mlr3)
library(mlr3learners)
library(MASS)

# Load data
data(default)

# Split data into training and test sets
set.seed(123)
task <- TaskClassif$new(id = "default", backend = default, target = "default")
splitter <- SplitterClassifHoldout$new(test_set_fraction = 0.2)
train_set <- splitter$train_set(task)
test_set <- splitter$test_set(task)

# Function to add noise to data and predict probabilities
noise_predict <- function(model, data, noise_level) {
  # Add noise to data
  noisy_data <- data + rnorm(nrow(data), mean = 0, sd = noise_level)
  
  # Predict probabilities
  predictions <- predict(model, newdata = noisy_data)
  probs <- as.data.frame(predictions$probabilities)
  
  return(probs)
}

# Estimate distribution of noise from data
sigma <- sd(unlist(train_set$X))
mu <- mean(unlist(train_set$X))
noise_dist <- fitdistr(unlist(train_set$X), "norm")

# Define base learners
lrn1 <- lrn("classif.logreg")
lrn2 <- lrn("classif.rpart")

# Define ensemble learner
lrn_ensemble <- lrn("ensemble.stacking", predictors = c(lrn1, lrn2))

# Fit model
model <- mlr_learn(lrn_ensemble, train_set)

# Simulate noise and predict probabilities for test set
probs <- noise_predict(model, test_set$X, noise_level = noise_dist$estimate[2])

# Estimate uncertainty level of predicted probabilities
uncertainty <- apply(probs, 1, sd)
