library(lme4)
library(readr)
library(dplyr)
library(ggplot2)

#data <- read_csv("/home/martin/Desktop/bouts/WT/danieau/bout_frequency.csv")
data <- read_csv("/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/DATA/WT/danieau/bout_frequency.csv")

data <- data %>%
  mutate(
    fish = factor(fish),
    day = factor(day),
    bout_category = factor(bout_category),
    bout_side = factor(bout_side),
    epoch_name = factor(epoch_name),
    stim_param = factor(stim_param)
  )

# data <- data %>%
#   filter(bout_frequency != 0)

# TODO maybe mirror bouts side x stim params and get rid of them?
# TODO maybe try to sketch what x/y plots you want to show 

# bout_freq vs trial time
# bout_freq vs trial_num
# % larva response vs trial num

model <- lmer(
  bout_frequency ~ trial_time  + trial_num + (trial_time + trial_num | epoch_name / stim_param) + (1 | fish) + (1 | bout_category),
  data = data
)

control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5))
model <- glmer(
  bout_counts ~ trial_time + offset(log(time_bin_duration)) + trial_num + (trial_time + trial_num | epoch_name / stim_param + bout_category) + (1 | fish),
  data = data,
  family = poisson,
  control = control
)

model <- lmer(
  bout_frequency ~ time_of_day_cos + time_of_day_sin + trial_num + trial_time +
    (1 | fish)  + 
    (trial_num | epoch_name / stim_param) + 
    (trial_time | epoch_name) + (trial_time | bout_category) + (trial_time | bout_side),
  data = data
)

summary(model)
coef(model)
anova(model)

## trial time

ggplot(data, aes(x = trial_time, y = bout_frequency, color= bout_category)) +
  geom_point() + geom_jitter() + facet_wrap(~ epoch_name) 


ggplot(data %>% filter(bout_category == "JT"), aes(x = trial_time, y = bout_frequency, color= bout_category)) +
  geom_point() + geom_jitter() + facet_wrap(~ epoch_name) 

## trial num

ggplot(data, aes(x = trial_num, y = bout_frequency, color= bout_category)) +
  geom_point() + geom_jitter() + facet_wrap(~ epoch_name)

ggplot(data %>% filter(bout_category == "JT"), aes(x = trial_num, y = bout_frequency, color= bout_category)) +
  geom_point() + geom_jitter() + facet_wrap(~ epoch_name)

### NOTE bout frequency might be over estimated on shorter time bins
### TODO Try zero-truncated Poisson / Quasi-Poisson / Negative binomial model?