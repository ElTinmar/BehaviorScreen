library(lme4)
library(readr)
library(dplyr)
library(ggplot2)

data <- read_csv("/home/martin/Desktop/bouts/WT/danieau/bout_frequency.csv")
data <- data %>%
  mutate(
    fish = factor(fish),
    day = factor(day),
    bout_category = factor(bout_category),
    bout_side = factor(bout_side),
    epoch_name = factor(epoch_name),
    stim_param = factor(stim_param)
  )


data_nonzero <- data %>%
  filter(bout_frequency != 0)

# TODO maybe mirror bouts side x stim params and get rid of them?
# TODO maybe try to sketch what x/y plots you want to show 

# bout_freq vs trial time
# bout_freq vs trial_num
# % larva response vs trial num

model <- lmer(
  bout_frequency ~ trial_time + trial_num + (1 + trial_time | epoch_name) + (1 + trial_num | epoch_name) + (1 | fish),
  data = data_nonzero
)

model <- glmer(
  bout_frequency ~ trial_time + trial_num + (1 + trial_time | epoch_name) + (1 + trial_num | epoch_name) + (1 | fish),
  data = data_nonzero,
  family = poisson
)

model <- lmer(
  bout_frequency ~ time_of_day_cos + time_of_day_sin + trial_num + trial_time +
    (1 | fish)  + 
    (1 + trial_num | epoch_name / stim_param) + 
    (1 + trial_time | epoch_name) + (1 + trial_time | bout_category) + (1 + trial_time | bout_side),
  data = data
)

summary(model)
anova(model)

## trial time

ggplot(data_nonzero, aes(x = trial_time, y = bout_frequency, color= bout_category)) +
  geom_point() + geom_jitter() + facet_wrap(~ epoch_name) 


ggplot(data_nonzero %>% filter(bout_category == "JT"), aes(x = trial_time, y = bout_frequency, color= bout_category)) +
  geom_point() + geom_jitter() + facet_wrap(~ epoch_name) 

## trial num

ggplot(data_nonzero, aes(x = trial_num, y = bout_frequency, color= bout_category)) +
  geom_point() + geom_jitter() + facet_wrap(~ epoch_name)

ggplot(data_nonzero %>% filter(bout_category == "JT"), aes(x = trial_num, y = bout_frequency, color= bout_category)) +
  geom_point() + geom_jitter() + facet_wrap(~ epoch_name)

### NOTE bout frequency might be over estimated on shorter time bins
### TODO Try zero-truncated Poisson / Quasi-Poisson / Negative binomial model?