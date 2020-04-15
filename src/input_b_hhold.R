library(rpart)

data_train = read.csv("../data/B_hhold_train.csv", header = TRUE)
data_test = read.csv("../data/B_hhold_test.csv", header = TRUE)


data_train = within(data_train, rm("IrxBnWxE"))  # more than 90 % of missing data
data_test = within(data_test, rm("IrxBnWxE"))

col_missing_data = colnames(data_train)[colSums(is.na(data_train)) > 0]
print(col_missing_data)

data_train_inputed = data_train
data_test_inputed = data_test


print("Train")
for(col in col_missing_data){
  print(col)
  class_mod = rpart(data_train[,col] ~ ., data=data_train[,!names(data_train) %in% "poor"], method="class", na.action=na.omit)
  class_predicted_probs <- predict(class_mod, data_train)
  class_predicted = unlist(lapply(colnames(class_predicted_probs)[max.col(class_predicted_probs, ties.method="random")], as.integer))
  data_train_inputed[,col] = class_predicted
}

print("Test")
for(col in col_missing_data){
  print(col)
  class_mod = rpart(data_test[,col] ~ ., data=data_test[,!names(data_test) %in% "poor"], method="class", na.action=na.omit)
  class_predicted_probs <- predict(class_mod, data_test)
  class_predicted = unlist(lapply(colnames(class_predicted_probs)[max.col(class_predicted_probs, ties.method="random")], as.integer))
  data_test_inputed[,col] = class_predicted
}


write.csv(data_train_inputed, "../data_inputed/B_hhold_train.csv")
write.csv(data_train_inputed, "../data_inputed/B_hhold_test.csv")
