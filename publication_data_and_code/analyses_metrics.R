library(brms)
library(ggplot2)
library(magrittr)
library(dplyr)
library(purrr)
library(forcats)
library(tidyr)
library(ggdist)
library(tidybayes)
library(ggplot2)
library(cowplot)
library(rstan)
library(brms)
library(ggrepel)
library(RColorBrewer)
library(gganimate)
library(posterior)
library(modelr)
library(scales)
library(comprehenr)
library(latex2exp)

#library(brmstools)
library(knitr)
library(kableExtra)
library(gt)
library(tidyverse)
library(sjPlot)
library(insight)
library(httr)

repo_directory <- "Desktop" 
results_folder <- paste(repo_directory, "debussy_piano/publication_data_and_code/results", sep="/")
debussy <- read.csv(paste(results_folder, "results.csv", sep="/"))


# Preprocessing --------------------------------------------------------
### Standardization

debussy$percentage_resonances_1_scaled <- scale(debussy$percentage_resonances_1)
debussy$percentage_resonances_2_scaled <- scale(debussy$percentage_resonances_2)
debussy$percentage_resonances_3_scaled <- scale(debussy$percentage_resonances_3)
debussy$percentage_resonances_4_scaled <- scale(debussy$percentage_resonances_4)
debussy$percentage_resonances_5_scaled <- scale(debussy$percentage_resonances_5)
debussy$percentage_resonances_6_scaled <- scale(debussy$percentage_resonances_6)

debussy$percentage_resonances_weighted_6_scaled <- scale(debussy$percentage_resonances_6)
debussy$percentage_resonances_weighted_5_scaled <- scale(debussy$percentage_resonances_5)
debussy$percentage_resonances_weighted_4_scaled <- scale(debussy$percentage_resonances_4)
debussy$percentage_resonances_weighted_3_scaled <- scale(debussy$percentage_resonances_3)
debussy$percentage_resonances_weighted_2_scaled <- scale(debussy$percentage_resonances_2)
debussy$percentage_resonances_weighted_1_scaled <- scale(debussy$percentage_resonances_1)

debussy$partition_entropy_scaled <- scale(debussy$partition_entropy)
debussy$inverse_coherence_scaled <- scale(debussy$inverse_coherence)

debussy$center_of_mass_6_scaled <- scale(debussy$percentage_resonances_6)

debussy$length_qb_scaled <- scale(debussy$length_qb)

debussy$year_scaled <- scale(debussy$year)

debussy$EarlyLate <- factor(debussy$year_scaled <= 0, ordered = TRUE, levels = c(TRUE, FALSE))

### Factors
# debussy$intYear <- factor(debussy$intYear, ordered = TRUE)
debussy$intYear <- factor(debussy$year_scaled, ordered = TRUE)

  
  

# Analysis ----------------------------------------------------------------
## Prior ----------------------------------------------------------------
Prior <- 
  c(prior(student_t(3, 0, 1), class = "b"))


#MODEL

# Fragmentation
Debussy_Model_Fragmentation <- brm(partition_entropy_scaled ~ year_scaled*length_qb_scaled ,
                                          data = debussy, 
                                          prior = Prior,
                                          inits = 0, 
                                          family = gaussian(),
                                          warmup = 1000, 
                                          iter = 10000, 
                                          chains = 4,
                                          core = 4
)

saveRDS(Debussy_Model_Fragmentation, paste(results_folder, "Debussy_Model_Fragmentation.rds", sep="/"))

Debussy_Model_Fragmentation <- readRDS(paste(results_folder, "Debussy_Model_Fragmentation.rds", sep="/"))

s <- summary(Debussy_Model_Fragmentation)
s

tab_model(Debussy_Model_Fragmentation, file = paste(results_folder, "FragmentationSummary.html", sep="/"))


#Set labels
earliest <- min(debussy$year)
xticks_labels <- c(earliest, earliest+10, earliest + 20, earliest + 30, earliest + 40)
xticks <- (xticks_labels-mean(debussy$year))/(sd(debussy$year))

plot(conditional_effects(Debussy_Model_Fragmentation), points = TRUE, point_args = list(size = 1, width = 0.025), theme = theme_classic(base_size = 20))[[1]] + labs(x= 'Year', y = 'H'~(sigma)) + scale_x_continuous( breaks = xticks, labels = xticks_labels)
                                                                                                                                                        
                                                                                                                                          
h <- hypothesis(Debussy_Model_Fragmentation, c('year_scaled>0', 'length_qb_scaled>0', 'year_scaled:length_qb_scaled>0'))
h


# Coherence

Debussy_Model_Coherence <- brm(inverse_coherence_scaled ~ year_scaled*length_qb_scaled ,
                                   data = debussy, 
                                   prior = Prior,
                                   inits = 0, 
                                   family = gaussian(),
                                   warmup = 1000, 
                                   iter = 10000, 
                                   chains = 4,
                                   core = 4
                                   
)

saveRDS(Debussy_Model_Coherence, paste(results_folder, "Debussy_Model_Coherence_31082022.rds", sep="/"))
Debussy_Model_Coherence <- readRDS(paste(results_folder, "Debussy_Model_Coherence_31082022.rds", sep="/"))

plot(conditional_effects(Debussy_Model_Coherence), points = TRUE)

summary(Debussy_Model_Coherence)

tab_model(Debussy_Model_Coherence, file = paste(results_folder, "CoherenceSummary.html", sep="/"))

classify_length <- function(x, threshold = 0.5) {
  if(x>threshold) {return('1')}
  else if(x< -1 * threshold) {return('-1')}
  else return('0')
}

debussy$classified_length <- apply(debussy$length_qb_scaled, 1, classify_length)

shortest <- min(debussy$length_qb)
size_ticks_labels <- c(100, 400, 700, 1000)
size_ticks <- (size_ticks_labels-mean(debussy$length_qb))/(sd(debussy$length_qb))

plot(conditional_effects(Debussy_Model_Coherence, effects = "year_scaled"), line_arg = c(alpha = 0.15), theme = theme_classic(base_size = 20))[[1]] + 
  geom_jitter(data = debussy, mapping = aes(x = year_scaled, y = inverse_coherence_scaled, size = length_qb_scaled), inherit.aes = FALSE, width = 0.04) + 
  labs(x= 'Year', y = 'P'~(sigma), color = 'Length'~(sigma), fill = 'Length'~(sigma), size = 'Length'~(qb)) +
  scale_x_continuous( breaks = xticks, labels = xticks_labels) +
  scale_size(range = c(0.5,4), breaks =size_ticks , labels = size_ticks_labels) 


h <- hypothesis(Debussy_Model_Coherence, c('year_scaled>0', 'length_qb_scaled>0', 'year_scaled:length_qb_scaled>0'))
h

tab = h$hypothesis #%>% select(-Star)
a = map_chr(tab, ~ifelse(class(.x)=="numeric", "r","l"))
tab = tab %>% 
  mutate(across(where(is.numeric), ~comma(., accuracy=0.01))) %>% 
  rename_all(~gsub("\\.", " ", .))
formatted_tab <- tab %>% 
  kbl(caption="Hypotheses tests", align=a) %>% 
  kable_classic(full_width=FALSE, html_font="Times New Roman")
save_kable(formatted_tab, paste(results_folder, "CoherenceHypotheses.html", sep="/"))

######
#MELTED DATA
#####

melted_df <- read.csv(paste(results_folder, "results_melted.csv", sep="/"))

# Preprocessing
melted_df$value_resonance

#Scale continuous variables
melted_df$value_resonance_scaled <- scale(as.double(melted_df$value_resonance))
melted_df$value_resonance_entropy_scaled <- scale(melted_df$value_resonance_entropy)
melted_df$value_inertia_scaled <- scale(melted_df$value_inertia)
melted_df$partition_entropy_scaled <- scale(melted_df$partition_entropy)

melted_df$year_scaled <- scale(melted_df$year)
melted_df$length_qb_scaled <- scale(melted_df$length_qb)

#Categorical variables as factors

melted_df$variable_resonance <- relevel(factor(melted_df$variable_resonance, ordered = FALSE), ref = 'percentage_resonances_5')
melted_df$variable_resonance_entropy <- relevel(factor(melted_df$variable_resonance_entropy, ordered = FALSE), ref = 'percentage_resonances_entropy_5')
melted_df$variable_inertia <- relevel(factor(melted_df$variable_inertia, ordered = FALSE), ref = 'moments_of_inertia_5')


# Analysis ----------------------------------------------------------------
## Prior ----------------------------------------------------------------
Prior <- 
  c(prior(student_t(3, 0, 1), class = "b"))

#MODEL

# %Resonance
Debussy_Model_Melted <- brm(value_resonance_scaled ~ variable_resonance*year_scaled*length_qb_scaled + (1+length_qb_scaled|fname),
                     data = melted_df, 
                     prior = Prior,
                     inits = 0, 
                     family = gaussian(),
                     warmup = 1000, 
                     iter = 10000, 
                     chains = 4,
                     core = 4
)

summary(Debussy_Model_Melted)

hypothesis(Debussy_Model_Melted, hypothesis = c('year_scaled < 0', 
                                                'year_scaled + variable_resonancepercentage_resonances_6:year_scaled > 0',
                                                'year_scaled + variable_resonancepercentage_resonances_4:year_scaled > 0', 
                                                'year_scaled + variable_resonancepercentage_resonances_3:year_scaled > 0',
                                                'year_scaled + variable_resonancepercentage_resonances_2:year_scaled > 0',
                                                'year_scaled + variable_resonancepercentage_resonances_1:year_scaled > 0'
                                                ) )


# %Resonance weighted by entropy

Debussy_Model_Melted_entropy <- brm(value_resonance_entropy_scaled ~ variable_resonance_entropy*year_scaled*length_qb_scaled + (1 + length_qb_scaled | fname),
                            data = melted_df, 
                            prior = Prior,
                            inits = 0, 
                            family = gaussian(),
                            warmup = 1000, 
                            iter = 3000, 
                            chains = 4,
                            core = 4
)

summary(Debussy_Model_Melted_entropy)

tab_model(Debussy_Model_Melted_entropy, file = paste(results_folder, "PrevalenceSummary.html", sep="/"))

### Line plot coefficients 4 & 5
cond_effects <- conditional_effects(Debussy_Model_Melted_entropy, effects = 'year_scaled:variable_resonance_entropy', points = TRUE)

interaction <- cond_effects$`year_scaled:variable_resonance_entropy`

interaction_only45 <- interaction[interaction$variable_resonance_entropy %in% c('percentage_resonances_entropy_5', 'percentage_resonances_entropy_4'), ]

interaction_only45 

ggplot(interaction_only45, aes(x = year_scaled, y = estimate__, color = variable_resonance_entropy)) + scale_color_manual(labels = c("5 (Diatonic)", '4 (Octatonic)'), values = c("#f8766d", "#00b0f6")) + #c("#f8766d", "#00b0f6")) +
  geom_smooth(method = 'loess', formula = 'y ~ x') + 
  geom_ribbon( aes(ymin = lower__, ymax = upper__, fill = variable_resonance_entropy, color = NULL), alpha = .15) + scale_fill_manual(labels = c("5 (Diatonic)", '4 (Octatonic)'), values = c("#f8766d", "#00b0f6")) +
  #geom_jitter(data = Debussy_Model_Melted_entropy$data[Debussy_Model_Melted_entropy$data$coefficient_resonance_entropy %in% c('percentage_resonances_entropy_5', 'percentage_resonances_entropy_4'), ], aes(x = year_scaled, y = value_resonance_entropy_scaled), width = 0.04, size = 0.8, alpha = 1) +
  theme_classic(base_size = 20) + xlab('Year') + ylab('W'~(sigma)) + scale_x_continuous( breaks = xticks, labels = xticks_labels) + 
  labs(color = 'DFT Component', fill = "DFT Component") #+ theme(legend.position = c(0.18, 0.94), 
                                                                #legend.background = element_rect(fill= "transparent"), 
                                                                #legend.key.height= unit(0.5, 'cm'),
                                                                #legend.key.width= unit(0.5, 'cm'))


## Posterior distributions all coefficients

SpreadDraws5 <- Debussy_Model_Melted_entropy %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_resonance_entropypercentage_resonances_entropy_1:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_2:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_3:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_4:year_scaled`,  
               `b_variable_resonance_entropypercentage_resonances_entropy_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled)

SpreadDraws5$Component <- '5'



SpreadDraws1 <- Debussy_Model_Melted_entropy %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_resonance_entropypercentage_resonances_entropy_1:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_2:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_3:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_4:year_scaled`,  
               `b_variable_resonance_entropypercentage_resonances_entropy_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_resonance_entropypercentage_resonances_entropy_1:year_scaled`)
SpreadDraws1$Component <- '1'


SpreadDraws2 <- Debussy_Model_Melted_entropy %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_resonance_entropypercentage_resonances_entropy_1:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_2:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_3:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_4:year_scaled`,  
               `b_variable_resonance_entropypercentage_resonances_entropy_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_resonance_entropypercentage_resonances_entropy_2:year_scaled`)
SpreadDraws2$Component <- '2'

SpreadDraws3 <- Debussy_Model_Melted_entropy %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_resonance_entropypercentage_resonances_entropy_1:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_2:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_3:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_4:year_scaled`,  
               `b_variable_resonance_entropypercentage_resonances_entropy_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_resonance_entropypercentage_resonances_entropy_3:year_scaled`)
SpreadDraws3$Component <- '3'

SpreadDraws4 <- Debussy_Model_Melted_entropy %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_resonance_entropypercentage_resonances_entropy_1:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_2:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_3:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_4:year_scaled`,  
               `b_variable_resonance_entropypercentage_resonances_entropy_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_resonance_entropypercentage_resonances_entropy_4:year_scaled`)
SpreadDraws4$Component <- '4'


SpreadDraws6 <- Debussy_Model_Melted_entropy %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_resonance_entropypercentage_resonances_entropy_1:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_2:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_3:year_scaled`, 
               `b_variable_resonance_entropypercentage_resonances_entropy_4:year_scaled`,  
               `b_variable_resonance_entropypercentage_resonances_entropy_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_resonance_entropypercentage_resonances_entropy_6:year_scaled`)
SpreadDraws6$Component <- '6'

SpreadDraws <- rbind(SpreadDraws1, SpreadDraws2, SpreadDraws3, SpreadDraws4, SpreadDraws5, SpreadDraws6)

SpreadDraws %>%
  ggplot(aes(y = Component, x = variable_mean, fill = stat(x > 0))) + 
  stat_halfeye(scale = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c("gray80", "skyblue")) + theme_classic(base_size = 20) +xlab(TeX("\\beta($k:y$)") )+ ylab('DFT Component (k)') + 
  labs(fill = TeX("\\beta>0")) #+
  #theme(legend.position = c(0.25, 0.22), 
          #legend.background = element_rect(fill= "transparent"), 
          #legend.key.height= unit(0.5, 'cm'),
          #legend.key.width= unit(0.5, 'cm'))



# Hypotheses tests

h <- hypothesis(Debussy_Model_Melted_entropy, hypothesis = c('year_scaled < 0', 
                                                             'year_scaled + variable_resonance_entropypercentage_resonances_entropy_6:year_scaled > 0',
                                                             'year_scaled + variable_resonance_entropypercentage_resonances_entropy_4:year_scaled > 0', 
                                                             'year_scaled + variable_resonance_entropypercentage_resonances_entropy_3:year_scaled > 0',
                                                             'year_scaled + variable_resonance_entropypercentage_resonances_entropy_2:year_scaled > 0',
                                                             'year_scaled + variable_resonance_entropypercentage_resonances_entropy_1:year_scaled > 0',
                                                             'variable_resonance_entropypercentage_resonances_entropy_6:year_scaled > 0',
                                                             'variable_resonance_entropypercentage_resonances_entropy_4:year_scaled > 0',
                                                             'variable_resonance_entropypercentage_resonances_entropy_3:year_scaled > 0',
                                                             'variable_resonance_entropypercentage_resonances_entropy_2:year_scaled > 0',
                                                             'variable_resonance_entropypercentage_resonances_entropy_1:year_scaled > 0',
                                                             'length_qb_scaled<0',
                                                             'length_qb_scaled + variable_resonance_entropypercentage_resonances_entropy_1:length_qb_scaled < 0',
                                                             'length_qb_scaled + variable_resonance_entropypercentage_resonances_entropy_2:length_qb_scaled < 0',
                                                             'length_qb_scaled + variable_resonance_entropypercentage_resonances_entropy_3:length_qb_scaled < 0',
                                                             'length_qb_scaled + variable_resonance_entropypercentage_resonances_entropy_4:length_qb_scaled < 0',
                                                             'length_qb_scaled + variable_resonance_entropypercentage_resonances_entropy_6:length_qb_scaled < 0'
                                                            
) )

h



# Moment of inertia

Debussy_Model_Melted_inertia <- brm(value_inertia_scaled ~ variable_inertia*year_scaled*length_qb_scaled + (1 + length_qb_scaled | fname),
                                    data = melted_df, 
                                    prior = Prior,
                                    inits = 0, 
                                    family = gaussian(),
                                    warmup = 1000, 
                                    iter = 10000, 
                                    chains = 4,
                                    core = 4
                                    
)

saveRDS(Debussy_Model_Melted_inertia, paste(results_folder, "Inertia_model_31082022.rds", sep="/"))
Debussy_Model_Melted_inertia <- readRDS(paste(results_folder, "Inertia_model_31082022.rds", sep="/"))


summary(Debussy_Model_Melted_inertia)

tab_model(Debussy_Model_Melted_inertia, file = paste(results_folder, "InertiaSummary.html", sep="/"))

conditional_effects(Debussy_Model_Melted_inertia)

cond_effects <- conditional_effects(Debussy_Model_Melted_inertia, effects = 'year_scaled:variable_inertia', points = TRUE)

interaction <- cond_effects$`year_scaled:variable_inertia`

interaction_only45 <- interaction[interaction$variable_inertia %in% c('moments_of_inertia_5', 'moments_of_inertia_4'), ]

interaction_only45 

ggplot(interaction_only45, aes(x = year_scaled, y = estimate__, color = variable_inertia)) + scale_color_manual(labels = c("5", '4'), values = c("#f8766d", "#00b0f6")) +
  geom_smooth(method = 'loess', formula = 'y ~ x') + 
  geom_ribbon( aes(ymin = lower__, ymax = upper__, fill = variable_inertia, color = NULL), alpha = .15) + scale_fill_manual(labels = c("5", '4'), values = c("#f8766d", "#00b0f6")) +
  geom_jitter(data = Debussy_Model_Melted_inertia$data[Debussy_Model_Melted_inertia$data$variable_inertia %in% c('moments_of_inertia_5', 'moments_of_inertia_4'), ], aes(x = year_scaled, y = value_inertia_scaled), width = 0.05, size = 0.5, alpha = 1) +
  theme_classic() + xlab('Year') + ylab('I'~(sigma)) + scale_x_continuous( breaks = xticks, labels = xticks_labels) + 
  labs(color = 'DFT Component', fill = "DFT Component") + theme(legend.position = c(0.15, 0.9), 
                                                              legend.background = element_rect(fill= "transparent"), 
                                                              legend.key.height= unit(0.5, 'cm'),
                                                              legend.key.width= unit(0.5, 'cm'))

## Posterior distributions all coefficients
SpreadDraws5 <- Debussy_Model_Melted_inertia %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_inertiamoments_of_inertia_1:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_2:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_3:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_4:year_scaled`,  
               `b_variable_inertiamoments_of_inertia_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled)

SpreadDraws5$Component <- '5'



SpreadDraws1 <- Debussy_Model_Melted_inertia %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_inertiamoments_of_inertia_1:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_2:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_3:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_4:year_scaled`,  
               `b_variable_inertiamoments_of_inertia_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_inertiamoments_of_inertia_1:year_scaled`)
SpreadDraws1$Component <- '1'


SpreadDraws2 <- Debussy_Model_Melted_inertia %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_inertiamoments_of_inertia_1:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_2:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_3:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_4:year_scaled`,  
               `b_variable_inertiamoments_of_inertia_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_inertiamoments_of_inertia_2:year_scaled`)
SpreadDraws2$Component <- '2'

SpreadDraws3 <- Debussy_Model_Melted_inertia %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_inertiamoments_of_inertia_1:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_2:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_3:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_4:year_scaled`,  
               `b_variable_inertiamoments_of_inertia_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_inertiamoments_of_inertia_3:year_scaled`)
SpreadDraws3$Component <- '3'

SpreadDraws4 <- Debussy_Model_Melted_inertia %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_inertiamoments_of_inertia_1:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_2:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_3:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_4:year_scaled`,  
               `b_variable_inertiamoments_of_inertia_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_inertiamoments_of_inertia_4:year_scaled`)
SpreadDraws4$Component <- '4'


SpreadDraws6 <- Debussy_Model_Melted_inertia %>%  
  spread_draws(b_Intercept, b_year_scaled, 
               `b_variable_inertiamoments_of_inertia_1:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_2:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_3:year_scaled`, 
               `b_variable_inertiamoments_of_inertia_4:year_scaled`,  
               `b_variable_inertiamoments_of_inertia_6:year_scaled`) %>%
  mutate(variable_mean =  b_year_scaled + `b_variable_inertiamoments_of_inertia_6:year_scaled`)
SpreadDraws6$Component <- '6'

SpreadDraws <- rbind(SpreadDraws1, SpreadDraws2, SpreadDraws3, SpreadDraws4, SpreadDraws5, SpreadDraws6)

SpreadDraws %>%
  ggplot(aes(y = Component, x = variable_mean, fill = stat(x > 0))) + 
  stat_halfeye(scale = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c("gray80", "skyblue")) + theme_classic(base_size = 16) +
  xlab(TeX("\\beta($k:y$)") )+ ylab('DFT Component (k)') + 
  labs(fill = TeX("\\beta>0")) #+




h <- hypothesis(Debussy_Model_Melted_inertia, hypothesis = c('year_scaled < 0', 
                                                             'year_scaled + variable_inertiamoments_of_inertia_6:year_scaled > 0',
                                                             'year_scaled + variable_inertiamoments_of_inertia_4:year_scaled > 0', 
                                                             'year_scaled + variable_inertiamoments_of_inertia_3:year_scaled > 0',
                                                             'year_scaled + variable_inertiamoments_of_inertia_2:year_scaled > 0',
                                                             'year_scaled + variable_inertiamoments_of_inertia_1:year_scaled > 0',
                                                             'variable_inertiamoments_of_inertia_6:year_scaled > 0',
                                                             'variable_inertiamoments_of_inertia_4:year_scaled > 0',
                                                             'variable_inertiamoments_of_inertia_3:year_scaled > 0',
                                                             'variable_inertiamoments_of_inertia_2:year_scaled > 0',
                                                             'variable_inertiamoments_of_inertia_1:year_scaled > 0',
                                                             'length_qb_scaled<0',
                                                             'length_qb_scaled + variable_inertiamoments_of_inertia_1:length_qb_scaled < 0',
                                                             'length_qb_scaled + variable_inertiamoments_of_inertia_2:length_qb_scaled < 0',
                                                             'length_qb_scaled + variable_inertiamoments_of_inertia_3:length_qb_scaled < 0',
                                                             'length_qb_scaled + variable_inertiamoments_of_inertia_4:length_qb_scaled < 0',
                                                             'length_qb_scaled + variable_inertiamoments_of_inertia_6:length_qb_scaled < 0'
) )

h

