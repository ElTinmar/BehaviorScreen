library(lme4)
library(readr)
library(dplyr)

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

# TODO maybe mirror bouts side x stim params and get rid of them?
model <- lmer(
  bout_frequency ~ time_of_day_cos + time_of_day_sin + trial_num + trial_time +
    (1 | fish)  + 
    (1 + trial_num | epoch_name) + 
    (1 + trial_time | epoch_name) + (1 + trial_time | bout_category) + (1 + trial_time | bout_side),
  data = data
)

summary(model)
anova(model)


ggplot(data, aes(x = bout_category, y = bout_frequency)) +
  geom_boxplot() +
  theme_classic()