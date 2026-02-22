# TODO maybe mirror bouts side x stim params and get rid of them?
# % larva response vs trial num
# % responsive trial  

# TODO cross validation?

library(lme4)
library(readr)
library(dplyr)
library(ggplot2)
library(mgcv)

bout_category_levels = c(
  "AS",
  "S1",
  "S2",
  "SCS",
  "LCS",
  "BS",
  "JT",
  "HAT",
  "RT",
  "SAT",
  "O",
  "LLC",
  "SLC"
)

## WT dataset ==============================================================================

#data <- read_csv("/home/martin/Desktop/bouts/WT/danieau/bout_frequency.csv")
#data <- read_csv("/home/martin/Downloads/Screen/WT/danieau/bout_frequency.csv")
#data <- read_csv("/media/martin/DATA1/Behavioral_screen/DATA/WT/danieau/bout_frequency.csv")
data <- read_csv("/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/DATA/Screen/WT/danieau/bout_frequency.csv")

data <- data %>%
  mutate(
    fish = factor(fish),
    day = factor(day),
    bout_category = factor(bout_category, levels=bout_category_levels),
    bout_side = factor(bout_side),
    epoch_name = factor(epoch_name),
    stim_param = factor(stim_param)
  )

data <- data %>%
  filter(!bout_category %in% c("SCS", "LCS")) %>%
  droplevels()

data$groups = interaction(data$epoch_name, data$stim_param, data$bout_category, data$bout_side, drop=TRUE)

data_trial_avg <- data %>%
  group_by(fish, dpf, day, time_of_day_cos, time_of_day_sin, epoch_name, stim_param, trial_time, bout_category, bout_side, groups) %>%
  summarize(
    bout_frequency = mean(bout_frequency, na.rm = TRUE),
    bout_counts = mean(bout_counts, na.rm = TRUE),
    .groups = "drop"
  )

## combined dataset ========================================================================

data <- read_csv("combined_bout_frequency.csv")

data <- data %>%
  mutate(
    line = factor(line),
    condition = factor(condition),
    fish = factor(fish),
    day = factor(day),
    bout_category = factor(bout_category, levels=bout_category_levels),
    bout_side = factor(bout_side),
    epoch_name = factor(epoch_name),
    stim_param = factor(stim_param)
  )
data <- data %>%
  filter(!bout_category %in% c("SCS", "LCS")) %>%
  droplevels()

data$groups = interaction(data$line, data$condition, data$epoch_name, data$stim_param, data$bout_category, data$bout_side, drop=TRUE)

data_trial_avg <- data %>%
  group_by(line, condition, fish, dpf, day, time_of_day_cos, time_of_day_sin, epoch_name, stim_param, trial_time, bout_category, bout_side, groups) %>%
  summarize(
    bout_frequency = mean(bout_frequency, na.rm = TRUE),
    bout_counts = mean(bout_counts, na.rm = TRUE),
    .groups = "drop"
  )

## subtract WT mean across trial x fish to get the proper baseline 
wt_avg <- data_trial_avg %>%
  filter(line == "WT", condition == "danieau") %>%
  group_by(epoch_name, stim_param, bout_category, bout_side, trial_time) %>%
  summarize(wt_bout_frequency = mean(bout_frequency), .groups = "drop")

data_comp <- data_trial_avg %>%
  left_join(wt_avg, by = c("epoch_name", "stim_param", "bout_category", "bout_side", "trial_time")) %>%
  mutate(delta_bout_frequency = bout_frequency - wt_bout_frequency) %>%
  filter(!(line == "WT" & condition == "danieau"))

## Checking distributions
# data_comp <- data_comp %>% filter(!line=="1010Kaede-X-81C")

hist(wt_avg$wt_bout_frequency, breaks = 200, prob = TRUE, col = rgb(0,0,1,0.5))
hist(data_comp$bout_frequency, breaks = 200, prob = TRUE, col = rgb(1,0,0,0.5), add=TRUE)

hist(data_comp$delta_bout_frequency, breaks= 200)
data_comp_treated <- data_comp %>% filter(condition=="ronidazole")
data_comp_untreated <- data_comp %>% filter(!condition=="ronidazole")

hist(data_comp_treated$delta_bout_frequency, breaks= 200, prob = TRUE, col = rgb(1,0,0,0.5))
hist(data_comp_untreated$delta_bout_frequency, breaks= 200, prob = TRUE, col = rgb(0,0,1,0.5), add=TRUE)

##### LM =====================================================================================

model <- lm(
  bout_frequency ~ groups,
  data = data
)

model <- lm(
  bout_frequency ~ groups,
  data = data_trial_avg
)

model <- lm(
  bout_frequency ~ trial_time * groups,
  data = data
)

model <- lm(
  bout_frequency ~ 0 + groups + trial_time:groups,
  data = data
)

model <- lm(
  bout_frequency ~ 0 + groups + trial_time:groups,
  data = data_trial_avg
)

model <- lm(
  bout_frequency ~ 0 + groups + trial_time:groups + trial_num:groups,
  data = data
)

##### GAM / GAMM ===============================================================================

model <- bam(
  bout_frequency ~ 0 + groups + s(trial_time, by=groups, k=10), 
  method = "fREML", 
  data = data,
  discrete = TRUE,
  nthreads = 20,
)

model <- bam(
  bout_frequency ~ 0 + groups + s(trial_time, by=groups) + s(trial_num, by=groups), 
  method = "fREML", 
  data = data,
  discrete = TRUE,
  nthreads = 20,
)

model <- bam(
  bout_frequency ~ 0 + groups + s(trial_time, by=groups, k=10), 
  method = "fREML", 
  data = data_trial_avg,
  discrete = TRUE,
  nthreads = 20,
)

model <- bam(
  bout_frequency ~ 0 + groups + s(trial_time, by=groups, k=10) + s(fish, bs = "re"),
  method = "fREML", 
  data = data_trial_avg,
  discrete = TRUE,
  nthreads = 20,
)

##### LMM =====================================================================================

model <- lmer(
  bout_frequency ~ groups + (1 | fish),
  data = data
)

model <- lmer(
  bout_frequency ~ trial_time * groups + (1 | fish),
  data = data
)

model <- lmer(
  bout_frequency ~ 0 + groups + trial_time:groups + (1 | fish),
  data = data
)

model <- lmer(
  bout_frequency ~ 0 + groups + trial_time:groups + (1 | fish),
  data = data_trial_avg
)

##### GLM =====================================================================================

model <- glm(
  bout_counts ~ groups + offset(log(time_bin_duration)),
  family = poisson,
  data = data
)

model <- glm(
  bout_counts ~ 0 + groups + trial_time:groups + offset(log(time_bin_duration)),
  family = poisson,
  data = data
)

model <- glm(
  bout_counts ~ 0 + groups + trial_num:groups + offset(log(time_bin_duration)),
  family = poisson,
  data = data
)

model <- glm(
  bout_counts ~ 0 + groups + trial_time:groups + trial_num:groups + offset(log(time_bin_duration)),
  family = poisson,
  data = data
)

model <- glm(
  bout_counts ~ trial_time * groups + offset(log(time_bin_duration)),
  family = poisson,
  data = data
)
deviance(model) / df.residual(model)

model <- glm(
  bout_counts ~ (trial_time + trial_num) * groups + offset(log(time_bin_duration)),
  family = poisson,
  data = data
)

model <- glm(
  bout_counts ~ 0 + groups + trial_time:groups + offset(log(time_bin_duration)),
  family = quasipoisson,
  data = data
)

##### GLMM =====================================================================================

model <- glmer(
  bout_counts ~ trial_time * groups + offset(log(time_bin_duration)) + (1 | fish),
  data = data,
  family = poisson,
)

model <- glmer(
  bout_counts ~ 0 + groups + trial_time:groups + offset(log(time_bin_duration)) + (1 | fish),
  data = data,
  family = poisson,
)

#####  Coefficients, diagnostics ===============================================================
# choose the right version depending on model:

# frequency modelled directly
data$pred_frequency <- fitted(model)
data$pred_frequency <- predict(model, newdata = data)

data_trial_avg$pred_frequency <- fitted(model)
data_trial_avg$pred_frequency <- predict(model, newdata = data_trial_avg)

# counts with poisson / quasipoisson / negative binomial
data$pred_count <- fitted(model)
data$pred_frequency <- data$pred_count / data$time_bin_duration

summary(model)
exp(coef(model))
anova(model)

# residuals 
ggplot(data, aes(x = residuals(model, type="response"))) +
  geom_histogram(binwidth = 0.1, alpha = 0.5) +
  labs(x = "Response residuals", y = "Count") +
  xlim(-2.5, 2.5)

ggplot(data_trial_avg, aes(x = residuals(model, type="response"))) +
  geom_histogram(binwidth = 0.1, alpha = 0.5) +
  labs(x = "Response residuals", y = "Count") 

plot(fitted(model),  residuals(model, type="response"))

##### plots ====================================================================================

# trial time
ggplot(data, aes(x = trial_time, y = bout_frequency)) +
  geom_point(alpha = 0.4) + 
  geom_jitter() + 
  geom_line(aes(x = trial_time, y = pred_frequency, color = bout_category), linewidth = 1.2) +
  facet_grid(bout_category*bout_side ~ epoch_name*stim_param)  

ggplot(data_trial_avg, aes(x = trial_time, y = bout_frequency)) +
  geom_point(alpha = 0.4) + 
  geom_jitter() + 
  geom_line(aes(x = trial_time, y = pred_frequency, color = bout_category), linewidth = 1.2) +
  facet_grid(bout_category*bout_side ~ epoch_name*stim_param)  

# trial num
ggplot(data, aes(x = trial_num, y = bout_frequency)) +
  geom_point(alpha = 0.4) +  
  geom_jitter() + 
  geom_line(aes(x = trial_num, y = pred_frequency, color = bout_category), linewidth = 1.2) +
  facet_grid(bout_category*bout_side ~ epoch_name*stim_param) 

