library(lme4)
library(readr)
library(dplyr)
library(ggplot2)

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

#data <- read_csv("/home/martin/Desktop/bouts/WT/danieau/bout_frequency.csv")
data <- read_csv("/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/DATA/Screen/WT/danieau/bout_frequency.csv")
#data <- read_csv("combined_bout_frequency.csv")
#data <- read_csv("/media/martin/DATA1/Behavioral_screen/DATA/WT/danieau/bout_frequency.csv")

data <- data %>%
  mutate(
    #line = factor(line),
    #condition = factor(condition),
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

data$groups = interaction(data$epoch_name, data$stim_param, data$bout_category, data$bout_side)

# TODO maybe mirror bouts side x stim params and get rid of them?
# TODO maybe try to sketch what x/y plots you want to show 

# bout_freq vs trial time
# bout_freq vs trial_num
# % larva response vs trial num
# % responsive trial  


##### Linear models =====================================================================================
model <- lm(
  bout_frequency ~ groups,
  data = data
)

model <- lm(
  bout_frequency ~ trial_time * groups,
  data = data
)

# dont use first level of groups as reference
model <- lm(
  bout_frequency ~ 0 + groups + trial_time:groups,
  data = data
)

model <- lm(
  bout_frequency ~ 0 + groups + trial_time:groups + trial_num:groups,
  data = data
)

## mixed effect

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

data$pred_count <- predict(model)   
data$pred_frequency <- data$pred_count / data$time_bin_duration

##### Poisson models =====================================================================================
model <- glm(
  bout_counts ~ groups + offset(log(time_bin_duration)),
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
###

model2 <- glm(
  bout_counts ~ (trial_time + trial_num) * groups + offset(log(time_bin_duration)),
  family = poisson,
  data = data
)

model <- glmer(
  bout_counts ~ trial_time * groups + offset(log(time_bin_duration)) + (1 | fish),
  data = data,
  family = poisson,
)
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

# model predictions

# trial time
ggplot(data, aes(x = trial_time, y = bout_frequency)) +
  geom_point(alpha = 0.4) + 
  geom_jitter() + 
  geom_line(aes(x = trial_time, y = pred_frequency, color = bout_category), linewidth = 1.2) +
  facet_grid(epoch_name*stim_param ~ bout_category*bout_side) 

# trial num
ggplot(data, aes(x = trial_num, y = bout_frequency)) +
  geom_point(alpha = 0.4) +  
  geom_jitter() + 
  geom_line(aes(x = trial_num, y = pred_frequency, color = bout_category), linewidth = 1.2) +
  facet_grid(epoch_name*stim_param ~ bout_category*bout_side) 

