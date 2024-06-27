# Kathryn Kingsley KLK170230 log regression
# Store data into a data frame. Explore data, and print graphs
df <- read.csv("titanic_project.csv")
summary(df)
names(df)
head(df)
tail(df)
cor(df[1:5], use="complete")
# pclass has the strongest correlation
boxplot(df$survived, df$pclass, col="red", xlab="Survived", ylab="Class", varwidth=TRUE)
boxplot(df$survived, df$age, col="pink", xlab="Survived", ylab="Age", varwidth=TRUE)
hist(df$age, col="blue", xlab="Age", main= "Passenger Ages")
cdplot(df$age, as.factor(df$sex), col=c("green","gray"), xlab="Survived", ylab="Age")

# Create logistic regression model and time it! 900 objects for test and 146 for train
train <- head(df, n=900)
test <- tail(df, n=146)
start_time <- Sys.time()
glm1 <- glm(survived~pclass, data = train, family = binomial)
end_time <- Sys.time()
final_time <- end_time - start_time
summary(glm1)

# extract coefficient
coef_class <- coef(glm1)["pclass"]
print(paste("Class coefficient: ", coef_class))

# test on test data
set.seed(1234)
probs <- predict(glm1, newdata = test, type="response")
pred <- ifelse(probs>0.5, 1, 0)

# calculate required totals
acc1 <- mean(pred==test$survived)
calc_table <- table(pred, test$survived)
calc_table
tp <- calc_table[1]
fn <- calc_table[2]
fp <- calc_table[3]
tn <- calc_table[4]
sens <- (tp/(tp+fn))
spec <- (tn/(tn+fp))

# print out
print(paste("Time taken: ", final_time))
print(paste("Accuracy: ", acc1))
print(paste("Specificity: ", spec))
print(paste("Sensitivity: ", sens))

