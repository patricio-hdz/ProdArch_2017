rm(list = ls(all.names = TRUE))
#Returns a string vector with the email domains of the participants in the list.
#@param path: a character variable with the path and name where the file containing the domains is.
#@return domains: a vector with the electronic domains.
if(require(stringr)==FALSE){
  install.packages("stringr")
}else
  library(stringr)
if(require("tm")==FALSE){
  install.packages("tm")
}else
  library(tm)
library(tm.plugin.dc)
domains <- function(path){
  doms <- str_split(readLines(path), pattern = " ")
  domains=character(length(doms))
  for(i in 1:length(doms))
    domains[i]=doms[[i]]  
  return(domains)
}

#Removes punctiation, number, english stopwords and words from the vector "removedWords", transform to lowercase and builds the DocumentTermMatrix.
#@param path: a character variable with the path containing the documents to be cleaned and converted into DocumentTermMatrix.
#@param removedWords: a character vector with the words that are of no interest for the analysis and you want to remove.
#@return dtm: a DocumentTermMatrix for the documents to be anaylized.
extropy.dtm <- function(path,removedWords){
  library(tm)
  library(doParallel)
  library(doSNOW)
  library(snowfall)
  cname <- file.path(path)
  filenames <- list.files(path,pattern="*.txt")
  toSpace <- content_transformer(function(x, pattern) { return (gsub(pattern, " " , x))})
  removehttp <- content_transformer(function(x,pattern) { return (gsub("http[A-Za-z]+", "", x))})
  removehtml <- content_transformer(function(x,pattern) { return (gsub("html[A-Za-z]+", "", x))})
  word.length <- content_transformer(function(x,pattern) { return ( gsub("\\b[[:alpha:]]{20,}\\b", "", x, perl=T))})
  #files <- lapply(filenames,readLines)
  docs <- Corpus(DirSource(cname))
  #writeLines(as.character(docs[[1]]))
  docs <- tm_map(docs, content_transformer(tolower))
  #remove potentially problematic symbols
  docs <- tm_map(docs, toSpace, "-")
  docs <- tm_map(docs, toSpace, "’")
  docs <- tm_map(docs, toSpace, "‘")
  docs <- tm_map(docs, toSpace, "•")
  docs <- tm_map(docs, content_transformer(removePunctuation))
  docs <- tm_map(docs, content_transformer(removeNumbers))
  docs <- tm_map(docs, word.length)
  docs <- tm_map(docs, removeWords, stopwords("english"))
  docs <- tm_map(docs, removeWords, removedWords)
  docs <- tm_map(docs,removehttp)
  docs <- tm_map(docs,removehtml)
  #docs <- tm_map(docs, stripWhitespace)
  #docs <- tm_map(docs,stemDocument)
  #writeLines(as.character(docs[[1]]))
  #docs <- tm_map(docs,stemDocument)
  #writeLines(as.character(docs[[1]]))
  dtm <- DocumentTermMatrix(docs) 
  sfInit(parallel=TRUE, cpus=detectCores(logical=TRUE)-1, type="SOCK") # for snowfall
  cl <- makeCluster(detectCores(logical=TRUE)-1, type = "SOCK") # for snow
  registerDoSNOW(cl) # for snow
  sfExport("dtm") # for snowfall
  sfLibrary(topicmodels) # for snowfall
  clusterEvalQ(cl, library(topicmodels)) # for snow
  clusterExport(cl, c("dtm")) # for snow
  rowTotals <- parApply(cl,dtm,1,sum)
  dtm.new   <- dtm[rowTotals> 0, ]
  stopCluster(cl)
  #rownames(dtm) <- filenames
  #inspect(dtm)
  #rowTotals <- apply(dtm , 1, sum) #Find the sum of words in each Document
  #empty.rows <- dtm[rowTotals == 0, ]$dimnames[1][[1]]
  #corpus <- docs[-as.numeric(empty.rows)]
  return(dtm.new)
  #dtm <- DocumentTermMatrix(docs) 
}
#Algunas palabras que deseamos remover.
myStopwords <- c("can", "say","one","way","use",
                  "also","howev","tell","will",
                  "much","need","take","tend","even",
                  "like","particular","rather","said",
                  "get","well","make","ask","come","end",
                  "first","two","help","often","may",
                  "might","see","someth","thing","point",
                  "post","look","right","now","think","‘ve ",
                  "‘re ","anoth","put","set","new","good",
                  "want","sure","kind","larg","yes,","day","etc",
                  "quit","sinc","attempt","lack","seen","awar",
                  "littl","ever","moreov","though","found","abl","don",
                  "enough","far","earli","away","achiev","draw",
                  "last","never","brief","bit","entir","brief","url","wrote",
                  "great","lot","http","mailgmailcom")
removedWords=c("{","}","will", "fwd","re","re: ","extropy","extropychat","extropy-chat","extropians","extrop")#,"httplistsextropyorgmailmanlistinfocgiextropychat")
path="~/Documentos/ArquitecturadeDatos/Proyecto/Modelos/Asuntos_Nuevos"
#Se ha visto que los dominios de los correos, por ejemplo gmail.com, aún permanecen y no son objeto de análisis por lo que son removidos.
dom=domains("~/Escritorio/direcciones.txt")
#guardamos la DocumentTermMatrix como objeto rds para que cada script que la necesite pueda cargarla posteriormente.
DTM=extropy.dtm(path,c(removedWords,myStopwords,dom))
saveRDS(extropy.dtm(path,c(removedWords,myStopwords,dom)),"dtm.rds")