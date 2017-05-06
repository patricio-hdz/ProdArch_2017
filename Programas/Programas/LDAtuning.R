library(tm)
library(ldatuning)
library(doParallel)
rm(list = ls(all.names = TRUE))
#Function that takes the perplexity measure to find the best k for the LDA model.
#@param dtm: an object of the type DocumentTermMatrix.
#@param ntopics: a numeric vector with the candidate(s) number of topics to be compared.
#@param plot: TRUE if want to plot the perplexity of the k's processed for all the metrics.
#@return k_estimada: the best k to use for the LDA model.
tuning_k <- function(dtm,ntopics,plot){
  tunes <- FindTopicsNumber(
    dtm,
    topics = ntopics,
    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010","Deveaud2014"),
    method = "Gibbs",
    control = list(seed = 77),
    mc.cores = detectCores(logical=TRUE)-1,
    verbose = TRUE
  )
  if(plot==TRUE)
    FindTopicsNumber_plot(tunes)
  k_estimada=sum(tunes$topics[which.max(tunes$Griffiths2004)],tunes$topics[which.max(tunes$Deveaud2014)],tunes$topics[which.min(tunes$CaoJuan2009)],tunes$topics[which.min(tunes$Arun2010)])/4
  return(round(k_estimada))
}

#Se carga el objeto dtm previamente creado por el script DocumentTermMatrix.R
dtm=readRDS("dtm.rds")
inspect(dtm)
dtm1=removeSparseTerms(dtm,0.9)
dtm2=removeSparseTerms(dtm,0.1)
inspect(dtm1)
inspect(dtm2)

rowTotals <- apply(dtm , 1, sum)
dtm.new   <- dtm[rowTotals> 0, ]
inspect(dtm.new)
#Se toman los números de tópicos a probar.
ntopics=c(2,3,4,5,6,7,8,9,10,seq(12,30,2))
tuning_k(dtm.new,ntopics,TRUE)


  