# Kathryn Kingsley KLK170230 naive bayes

# Store data into a data frame. Explore data, and print graphs
library(e1071) # required for Naive Bayes
df <- read.csv("titanic_project.csv")
df$pclass<-factor(df$pclass)
df$sex<-factor(df$sex)
df$survived<-factor(df$survived)

# Create naive bayes model and time it! 900 objects for test and 146 for train
# print the model
train <- head(df, n=900)
test <- tail(df, n=146)
start_time <- Sys.time()
nb1 <- naiveBayes(survived~pclass+sex+age, data = train)
end_time <- Sys.time()
final_time <- end_time - start_time
nb1

# test on the test data
set.seed(1234)
pred <- predict(nb1, newdata = test, type= "class")

# make calculations
calc_table<- table(pred, test$survived)
acc <- mean(pred==test$survived)
tp <- calc_table[1]
fn <- calc_table[2]
fp <- calc_table[3]
tn <- calc_table[4]
sens <- (tp/(tp+fn))
spec <- (tn/(tn+fp))

# print out
print(paste("Time taken: ", final_time))
print(paste("Accuracy: ", acc))
print(paste("Specificity: ", spec))
print(paste("Sensitivity: ", sens))



