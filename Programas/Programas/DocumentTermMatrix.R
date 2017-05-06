rm(list = ls(all.names = TRUE))
#Removes punctiation, number, english stopwords and words from the vector "removedWords", transform to lowercase and builds the DocumentTermMatrix.
#@param path: a character variable with the path containing the documents to be cleaned and converted into DocumentTermMatrix.
#@param removedWords: a character vector with the words that are of no interest for the analysis and you want to remove.
#@return dtm: a DocumentTermMatrix for the documents to be anaylized.
extropy.dtm <- function(path,removedWords){
library(tm)
cname <- file.path(path)
docs <- Corpus(DirSource(cname))
docs <- tm_map(docs, content_transformer(removePunctuation))
docs <- tm_map(docs, content_transformer(removeNumbers))
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removeWords, removedWords)
docs <- tm_map(docs, PlainTextDocument)
 docs <- Corpus(VectorSource(docs))
dtm <- DocumentTermMatrix(docs) 
#dtm <- removeSparseTerms(dtm,0.3)
#inspect(dtm)
return(dtm)
}
r=c("{","}","will", "fwd","re","re: ","extropy","extropychat","extropy-chat","extropians","extrop","httplistsextropyorgmailmanlistinfocgiextropychat")
path="~/Escritorio/exp"
saveRDS(extropy.dtm(path,r),"dtm.rds")


