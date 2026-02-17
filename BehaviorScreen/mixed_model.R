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
data <- read_csv("/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen/DATA/WT/danieau/bout_frequency.csv")

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

# TODO maybe mirror bouts side x stim params and get rid of them?
# TODO maybe try to sketch what x/y plots you want to show 

# bout_freq vs trial time
# bout_freq vs trial_num
# % larva response vs trial num
# % responsive trial  

model <- lmer(
  bout_frequency ~ trial_time  + trial_num + 
  (trial_time + trial_num | epoch_name / stim_param + bout_category),
  #+ (1 | fish),
  data = data
)

#control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5))
model <- glmer(
  bout_counts ~ trial_time + offset(log(time_bin_duration)) + trial_num + 
    (trial_time + trial_num | epoch_name / stim_param + bout_category) + 
    (1 | fish),
  data = data,
  family = poisson,
  #control = control
)

summary(model)
coef(model)
anova(model)

# trial time
ggplot(data, aes(x = trial_time, y = bout_frequency, color = bout_category)) +
  geom_point() + 
  geom_jitter() + 
  facet_grid(epoch_name*stim_param ~ bout_category*bout_side) 

# trial num
ggplot(data, aes(x = trial_num, y = bout_frequency, color = bout_category)) +
  geom_point() + 
  geom_jitter() + 
  facet_grid(epoch_name*stim_param ~ bout_category*bout_side) 

