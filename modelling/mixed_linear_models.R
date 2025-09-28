setwd("/Users/minminchan/Documents/projects/dropout_risk_performance_forecasting/")
library(lme4)

# Load the data
perf_data <- read.csv("data/course performance data.csv")
dropout_data <- read.csv("data/student enrollment data.csv")
dropout_data['dropout'] <- ifelse(dropout_data$enrollment_status == 'dropped out', 1, 0)

# merge perf_data with dropout data (dropped university_gpa and entry_year)
merged_data <- merge(perf_data, dropout_data[,c("student_id","dropout","gender","age","department","high_school_gpa")], by = c("student_id"))
# drop columns from course_id and course_name
merged_data <- merged_data[, !(names(merged_data) %in% c("course_id", "course_name"))]

merged_data$gender <- as.factor(merged_data$gender)
merged_data$department <- as.factor(merged_data$department)
merged_data$dropout <- as.factor(merged_data$dropout)
merged_data$grade <- as.factor(merged_data$grade)
merged_data$completion_status <- as.factor(merged_data$completion_status)
merged_data$semester <- as.factor(merged_data$semester)
merged_data$student_id <- as.factor(merged_data$student_id)
# Modelling starts
lmm_model <- glmer(dropout ~ gender+department+grade * semester + completion_status * semester + (1|student_id), control = glmerControl(optimizer ="Nelder_Mead"), data=merged_data, family = binomial)

# References:
# https://stats.stackexchange.com/questions/58745/using-lmer-for-repeated-measures-linear-mixed-effect-model