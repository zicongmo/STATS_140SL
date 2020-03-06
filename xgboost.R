data <- read.csv("airbnb_la.csv")

data <- data %>% select(-X) %>% mutate(host_is_superhost = from_boolean(host_is_superhost),
                                       host_has_profile_pic = from_boolean(host_has_profile_pic),
                                       host_identity_verified = from_boolean(host_identity_verified),
                                       is_location_exact = from_boolean(is_location_exact),
                                       require_guest_profile_picture = from_boolean(require_guest_profile_picture),
                                       require_guest_phone_verification = from_boolean(require_guest_phone_verification),
                                       requires_license = from_boolean(requires_license),
                                       instant_bookable = from_boolean(instant_bookable))

data <- one_hot(as.data.table(data))

write.csv(data, file="data.csv", row.names = FALSE)

df <- read.csv("data.csv")
set.seed(12345)
n <- nrow(df)
train_indices <- sample(1:n, n * 0.8)

train <- df[train_indices,]
test <- df[-train_indices,]

# column 343: log_price
train_x <- df[train_indices,-343]
train_y <- df[train_indices,]$log_price
test_x <- df[-train_indices, -343]
test_y <- df[-train_indices,]$log_price

# Baseline
# Test MSE: 0.7524372
mean((mean(test_y) - test_y)^2)

# Linear Regression
# Test MSE: 0.2014361
model_linear <- lm(log_price ~ ., data = train)
mean((predict(model_linear, newdata = test) - test$log_price)^2)

# Ridge Regression
# Test MSE: 0.2016365
cv_fit_ridge <- cv.glmnet(as.matrix(train_x), train_y, alpha = 0)
opt_lambda <- cv_fit_ridge$lambda.min
model_ridge <- glmnet(as.matrix(train_x), train_y, alpha = 0, lambda = opt_lambda)
mean((predict(model_ridge, newx = as.matrix(test_x), s = opt_lambda) - test_y)^2)

# Lasso Regression
# Test MSE: 0.2013439
cv_fit_lasso <- cv.glmnet(as.matrix(train_x), train_y, alpha = 1)
opt_lambda_lasso <- cv_fit_lasso$lambda.min
model_lasso <- glmnet(as.matrix(train_x), train_y, alpha = 1, lambda = opt_lambda_lasso)
mean((predict(model_lasso, newx = as.matrix(test_x), s = opt_lambda) - test_y)^2)

# xgboost
# random training round 1: seed 987654321
# etas: 0.01, 0.02, ..., 0.99
# gammas: 0, 1, ..., 10
# depths: 3, 4, ..., 20
# 0.76, 3, 3 => 37 iterations, 0.435886
# 0.56, 6, 7 => 13 iterations, 0.442426
# 0.66, 5, 9 => 13 iterations, 0.439146
# 0.29, 0, 6, => 100 iterations, 0.411896
# 0.4, 7, 19 => 20 iterations 0.434
# 0.55, 10, 7 => 15 iterations, 0.4483
# 0.16, 0, 11 => 100 iterations, 0.407
# 0.49, 10, 4 => 18 iterations, 0.4502
# 0.06, 1, 6 => 100 iterations, 0.430
# 0.49, 7, 19 => 16 iterations, 0.440497
# 0.41, 2, 14 => 21 iterations, 0.420317
# 0.64, 1, 7 = 41 iterations, 0.425024
# 0.36, 4, 10 => 24 iterations, 0.428153 
# 0.51, 2, 14 => 24 iterations, 0.425079
# 0.51, 0, 3, => 100 iterations, 0.420275
# 0.81, 7, 20 => 7 iterations, 0.443438
# 0.43, 10, 10 => 20 iterations, 0.446
# 0.17, 10, 9 => 54 iterations, 0.441614
# 0.02, 4, 5 => 100 iterations, 0.750
# 0.2, 3, 4 => 79 iterations, 0.4328

# best: 
# 0.16, 0, 11, 0.407

# random training round 2: seed 456
# etas: 0.01, 0.02, ..., 0.20
# gammas: 0, 1, 2, 3
# depths: 8, 9, 10, 11, 12, 13, 14, 15
# 0.13, 0, 10 => 100 iterations, 0.4069
# 0.2, 0, 15 => 100 iterations,, 0.4115
# 0.07, 1, 13 => 100 iterations, 0.41244
# 0.1, 0, 10 => 100 iterations, 0.41057
# 0.14, 1, 13 => 100 iterations, 0.409622
# 0.11, 2, 13 => 74 iterations, 0.417124
# 0.2, 0, 15 => 100 iterations, 0.412471
# 0.02, 3, 11 => 100 iterations, 0.7383 (probably does better, but eta too low)
# 0.19, 2, 10 => 46 iterations, 0.420094
# 0.15, 2, 14 => 75 iterations, 0.416
# 0.07, 1, 13 => 100 iterations, 0.4117
# 0.15, 3, 15 => 57 iterations 0.419981
# 0.01, 2, 9 => 100 iterations, 1.669
# 0.14, 0, 9 => 100 iterations, 0.4081
# 0.17, 1, 14 => 64 iterations, 0.4133
# 0.08, 0, 8 => 100 iterations, 0.4163
# 0.15, 3, 13, => 55 iterations, 0.4198
# 0.19, 1, 10 => 62 iterations, 0.4095
# 0.06, 0, 9 => 100 iterations, 0.4185
# 0.1, 3, 14 => 100 iterations, 0.42119
set.seed(456)
etas <- seq(0.01, 0.2, by = 0.01)
gammas <- 0:3
depths <- 8:15

num_models <- 20
for (i in 1:num_models) {
    random_eta <- sample(etas, 1)
    random_gamma <- sample(gammas, 1)
    random_depth <- sample(depths, 1)
    
    params <- list(eta = random_eta, gamma = random_gamma, max_depth = random_depth)
    
    print(c(random_eta, random_gamma, random_depth))
    xgbcv <- xgb.cv(param = params, data = as.matrix(train_x), label = train_y, nrounds = 100, nfold = 5, print_every_n = 20, early_stopping_rounds = 20, maximize = F)
}

# finding best iterations for our best models
# 0.16, 0, 11 => 148 iterations, 0.407
# 0.13, 0, 10 => 257 iterations, 0.40500
# 0.14, 1, 13 => 88 iterations, 0.4121
# 0.14, 0, 9 => 288 iterations, 0.4051
# 0.29, 0, 6 => 224 iterations, 0.40727
params_16_0_11 <- list(eta = 0.16, gamma = 0, max_depth = 11)
xgbcv_16_0_11 <- xgb.cv(param = params_16_0_11, data = as.matrix(train_x), label = train_y, nrounds = 500, nfold = 5, print_every_n = 20, early_stopping_rounds = 20, maximize = F)

params_13_0_10 <- list(eta = 0.13, gamma = 0, max_depth = 10)
xgbcv_13_0_10 <- xgb.cv(param = params_13_0_10, data = as.matrix(train_x), label = train_y, nrounds = 500, nfold = 5, print_every_n = 20, early_stopping_rounds = 20, maximize = F)

params_14_1_13 <- list(eta = 0.14, gamma = 1, max_depth = 13)
xgbcv_14_1_13 <- xgb.cv(param = params_14_1_13, data = as.matrix(train_x), label = train_y, nrounds = 500, nfold = 5, print_every_n = 20, early_stopping_rounds = 20, maximize = F)

params_14_0_9 <- list(eta = 0.14, gamma = 0, max_depth = 9)
xgbcv_14_0_9 <- xgb.cv(param = params_14_0_9, data = as.matrix(train_x), label = train_y, nrounds = 500, nfold = 5, print_every_n = 20, early_stopping_rounds = 20, maximize = F)

params_29_0_6 <- list(eta = 0.29, gamma = 0, max_depth = 6)
xbcv_29_0_6 <- xgb.cv(param = params_29_0_6, data = as.matrix(train_x), label = train_y, nrounds = 500, nfold = 5, print_every_n = 20, early_stopping_rounds = 20, maximize = F)

# Looks like our best model is eta = 0.13, gamma = 0, max_depth = 10, nrounds = 257
# There's likely a model that gets better cv error, but the search space is just too large
# Test MSE: 0.1546625
xgb <- xgboost(data = as.matrix(train_x), label = train_y, params = params_13_0_10, nrounds = 257, print_every_n = 20)
mean((predict(xgb, as.matrix(test_x)) - test_y)^2)
