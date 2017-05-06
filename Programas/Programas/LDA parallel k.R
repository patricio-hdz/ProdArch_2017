# Speed tests of different parallel and non-parallel methods
# for iterating over different numbers of topics with 
# topicmodels

# clear workspace and stop any previous cluster instances
rm(list = ls(all.names = TRUE))
gc()
sfStop() 
install.packages("topicmodels")
install.packages("plyr")
install.packages("snowfall")
install.packages("foreach")
install.packages("doSNOW")
install.packages("qdapTools")
install.packages("ngram")
install.packages("lda")
install.packages("tm")
install.packages("tm.plugin.dc")
install.packages("doParallel")




library(topicmodels)
library(plyr)
library(snowfall)
library(foreach)
library(doSNOW)
library(qdapTools)
library(ngram)
library(lda)
library(tm)
library(tm.plugin.dc)
library(doParallel)
# get data
dtm=readRDS("dtm.rds")
rowTotals <- parApply(cl,dtm,1,sum)
dtm.new   <- dtm[rowTotals> 0, ]
# set number of topics to start with
k <- 30

# set model options
control_LDA_VEM <-
  list(estimate.alpha = TRUE, alpha = 50/k, estimate.beta = TRUE,
       verbose = 0, prefix = tempfile(), save = 0, keep = 0,
       seed = as.integer(100), nstart = 1, best = TRUE,
       var = list(iter.max = 200, tol = 10^-6),
       em = list(iter.max = 200, tol = 10^-4),
       initialize = "random")



# set sequence of topic numbers to iterate over
seq <- seq(30,40,1)
seq
# set parallel processing options
rm(list = c("dtm","rowTotals"))
# initiate cores
sfInit(parallel=TRUE, cpus=detectCores(logical=TRUE)-1, type="SOCK") # for snowfall
cl <- makeCluster(detectCores(logical=TRUE)-1, type = "SOCK") # for snow
registerDoSNOW(cl) # for snow

# send data and packages to multicores 
sfExport("dtm.new", "control_LDA_VEM") # for snowfall
sfLibrary(topicmodels) # for snowfall

# again for snow
clusterEvalQ(cl, library(topicmodels)) # for snow
clusterExport(cl, c("dtm.new", "control_LDA_VEM")) # for snow

# parallel methods

# wrapper for some of these that don't handle the function well
wrapper <- function (d) topicmodels:::LDA(dtm, control = control_LDA_VEM, d)

# using plyr in parallel (needs snow options)
best.model.PLYRP <<- llply(seq, function(d){topicmodels:::LDA(dtm.new, control = control_LDA_VEM, d)}, .parallel = TRUE)
# inspect results
stopCluster(cl)
install.packages("tidytext")
library(tidytext)

ap_topics <- tidy(best.model.PLYRP[[7]], matrix = "beta")
ap_topics

install.packages("ggplot2")
install.packages("dplyr")
library(ggplot2)
library(dplyr)

ap_top_terms <- ap_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)


ap_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
