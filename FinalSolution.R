#####<-------------------------------TASK 01 ------------------------------------>
# Load the required libraries
library(e1071) # For SVM model
library(randomForest) # For random forest model
library(caret) 
library(lubridate) # for converting datatime data type

# Read the training and testing datasets
train_data <- read.csv("/Users/zarifmahmud/Desktop/Home Excercise 2/rentalTrain.csv", header = TRUE)
test_data <- read.csv("/Users/zarifmahmud/Desktop/Home Excercise 2/rentalTest.csv", header = TRUE)

combine_data <- rbind(train_data[, 1:10], test_data)



# Factorize train data
combine_data$weather <- as.factor(combine_data$weather)
combine_data$workingday <- as.factor(combine_data$workingday)
combine_data$holiday <- as.factor(combine_data$holiday)
combine_data$season <- as.factor(combine_data$season)

#checking the levels and data type for the features
str(combine_data)

train_newdata <- cbind(combine_data[1:8708, ], train_data[, 11:13])
train_newdata <- train_newdata[, !(names(train_data) %in% c("casual", "datetime", "registered", "X"))]
test_newdata <- combine_data[(nrow(train_data) + 1):nrow(combine_data), ]
test_newdata<-subset(test_newdata, select = !(names(test_data) %in% c("datetime","X")))


# Set the seed for reproducibility
set.seed(123)

# Define the number of folds for cross-validation
k <- 5

# Initialize empty vectors to store RMSE values
svm_rmse <- rep(0, k)
rf_rmse <- rep(0, k)
lm_rmse <- rep(0, k)

# Perform k-fold cross-validation
for (i in 1:k) {
  # Split the data into training and validation sets
  folds <- cut(seq(1, nrow(train_newdata)), breaks = k, labels = FALSE)
  validation_indexes <- which(folds == i)
  train_indexes <- which(folds != i)
  train_fold <- train_newdata[train_indexes, ]
  validation_fold <- train_newdata[validation_indexes, ]
  
  # Train the SVM model
  svm_model <- svm(count ~ ., data = train_fold)
  
  # Make predictions using the SVM model on the validation set
  svm_predictions <- predict(svm_model, newdata = validation_fold)
  
  # Calculate the RMSE for SVM
  svm_rmse[i] <- sqrt(mean((svm_predictions - validation_fold$count)^2))
  
  # Train the random forest model
  rf_model <- randomForest(count ~ ., data = train_fold)
  
  # Make predictions using the random forest model on the validation set
  rf_predictions <- predict(rf_model, newdata = validation_fold)
  
  # Calculate the RMSE for random forest
  rf_rmse[i] <- sqrt(mean((rf_predictions - validation_fold$count)^2))
  
  # Train the linear regression model
  lm_model <- lm(count ~ ., data = train_fold)
  
  # Make predictions using the linear regression model on the validation set
  lm_predictions <- predict(lm_model, newdata = validation_fold)
  
  # Calculate the RMSE for linear regression
  lm_rmse[i] <- sqrt(mean((lm_predictions - validation_fold$count)^2))
}

# Print the average RMSE for each model
cat("SVM RMSE:", mean(svm_rmse), "\n")
cat("Random Forest RMSE:", mean(rf_rmse), "\n")
cat("Linear Regression RMSE:", mean(lm_rmse), "\n")

# Train the models on the full training data
svm_model <- svm(count ~ ., data = train_newdata)
rf_model <- randomForest(count ~ ., data = train_newdata)
lm_model <- lm(count ~ ., data = train_newdata)

# Make predictions on the test data using the trained models
svm_predictions <- predict(svm_model, newdata = test_newdata)
rf_predictions <- predict(rf_model, newdata = test_newdata)
lm_predictions <- predict(lm_model, newdata = test_newdata)

# Compare the predictions
comparison <- data.frame(SVM = svm_predictions, RandomForest = rf_predictions, LinearRegression = lm_predictions)
write.csv(comparison, file = "predictions.csv", row.names = FALSE)

#<--------------------------------------------Task-02---------------------------->

# Combine the training and testing datasets
combined_data1 <- combine_data
new_combine_data<- combined_data1[, 7:10]

# Scale the data
scaled_data <- scale(new_combine_data)


# Perform k-means clustering
set.seed(123)  # For reproducibility
k <- 3  # Number of clusters
kmeans_model <- kmeans(new_combine_data, centers = k)

# Get the cluster assignments for each data point
cluster_assignments <- kmeans_model$cluster

# Add the cluster assignments back to the combined dataset
combined_data1$cluster <- cluster_assignments

# Evaluate if the clusters separate based on the target variable
cluster_summary <- aggregate(count ~ cluster, data = combined_data1, FUN = mean)

# Print the cluster summary
cat("Cluster Summary:\n")
print(cluster_summary)

