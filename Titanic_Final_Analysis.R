# WEEK 1 TASK: DATA CLEANING AND PRELIMINARY ANALYSIS WITH R
# Dataset: Titanic passenger dataset (891 observations, 15 variables)

# 1. PACKAGES
# Run once if needed:
# install.packages(c("tidyverse", "caret", "corrplot"), dependencies = TRUE)
library(tidyverse)
library(caret)
library(corrplot)

# 2. IMPORT DATA
url <- "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
titanic <- read.csv(url, stringsAsFactors = FALSE)

# 3. INITIAL INSPECTION
head(titanic)
str(titanic)
dim(titanic)
summary(titanic)

# 4. MISSING-VALUE ANALYSIS
colSums(is.na(titanic))
sum(titanic$deck == "")
sum(titanic$embarked == "")

# 5. DUPLICATE CHECK
sum(duplicated(titanic))

# 6. DATA CLEANING
titanic_clean <- titanic

# Median imputation for missing Age
age_median <- median(titanic_clean$age, na.rm = TRUE)
titanic_clean$age[is.na(titanic_clean$age)] <- age_median

# Mode treatment for blank Embarked values
mode_embarked <- names(sort(table(titanic_clean$embarked), decreasing = TRUE))[1]
titanic_clean$embarked[titanic_clean$embarked == ""] <- mode_embarked

# Preserve blank Deck information as an Unknown category
titanic_clean$deck[titanic_clean$deck == ""] <- "Unknown"

# 7. CATEGORICAL ENCODING
titanic_clean$sex <- factor(titanic_clean$sex)
titanic_clean$embarked <- factor(titanic_clean$embarked)
titanic_clean$pclass <- factor(titanic_clean$pclass)
titanic_clean$class <- factor(titanic_clean$class)
titanic_clean$who <- factor(titanic_clean$who)

str(titanic_clean[, c("sex", "embarked", "pclass", "class", "who")])
colSums(is.na(titanic_clean))

# 8. OUTLIER DETECTION USING IQR
iqr_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR_value <- IQR(x, na.rm = TRUE)
  lower <- Q1 - 1.5 * IQR_value
  upper <- Q3 + 1.5 * IQR_value
  which(x < lower | x > upper)
}

age_outliers <- iqr_outliers(titanic_clean$age)
fare_outliers <- iqr_outliers(titanic_clean$fare)
length(age_outliers)
length(fare_outliers)

# Boxplots
boxplot(titanic_clean$age, main = "Age Boxplot", ylab = "Age")
boxplot(titanic_clean$fare, main = "Fare Boxplot", ylab = "Fare")

# 9. NORMALIZATION / STANDARDIZATION
titanic_clean$age_scaled <- as.numeric(scale(titanic_clean$age))
titanic_clean$fare_scaled <- as.numeric(scale(titanic_clean$fare))
summary(titanic_clean$age_scaled)
summary(titanic_clean$fare_scaled)

# 10. DUMMY / ONE-HOT ENCODING
dummy_model <- dummyVars(~ sex + embarked + pclass, data = titanic_clean)
dummy_data <- predict(dummy_model, newdata = titanic_clean)
dummy_data <- as.data.frame(dummy_data)
head(dummy_data)

# 11. PRELIMINARY ANALYSIS
table(titanic_clean$survived)
prop.table(table(titanic_clean$survived))

sex_survival <- titanic_clean %>%
  group_by(sex) %>%
  summarise(Survival_Rate = mean(survived) * 100)
print(sex_survival)

class_survival <- titanic_clean %>%
  group_by(pclass) %>%
  summarise(Survival_Rate = mean(survived) * 100)
print(class_survival)

# 12. VISUALIZATIONS
ggplot(titanic_clean, aes(x = factor(survived))) +
  geom_bar() +
  labs(x = "Survived (0 = No, 1 = Yes)", y = "Number of Passengers",
       title = "Passenger Survival Count")

ggplot(titanic_clean, aes(x = sex, fill = factor(survived))) +
  geom_bar(position = "fill") +
  labs(x = "Sex", y = "Proportion", fill = "Survived",
       title = "Survival Proportion by Sex")

ggplot(titanic_clean, aes(x = pclass, fill = factor(survived))) +
  geom_bar(position = "fill") +
  labs(x = "Passenger Class", y = "Proportion", fill = "Survived",
       title = "Survival Proportion by Passenger Class")

ggplot(titanic_clean, aes(x = age)) +
  geom_histogram(bins = 30) +
  labs(title = "Age Distribution", x = "Age", y = "Number of Passengers")

ggplot(titanic_clean, aes(x = fare)) +
  geom_histogram(bins = 30) +
  labs(title = "Fare Distribution", x = "Fare", y = "Number of Passengers")

# 13. CORRELATION ANALYSIS
numeric_data <- titanic_clean %>%
  select(survived, age, sibsp, parch, fare)
cor_matrix <- cor(numeric_data)
print(cor_matrix)

corrplot(cor_matrix, method = "number", type = "upper",
         tl.col = "black", tl.srt = 45)

# 14. SAVE CLEANED DATA
write.csv(titanic_clean, "titanic_cleaned.csv", row.names = FALSE)
getwd()
file.exists("titanic_cleaned.csv")

# END OF WEEK 1 ANALYSIS
