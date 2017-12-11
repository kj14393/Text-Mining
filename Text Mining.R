
## Building a predictive model for movie review sentiment##
# Data source: http://www.cs.cornell.edu/people/pabo/movie-review-data/

install.packages(c('tm', 'SnowballC', 'wordcloud'))
library(tm)
library(SnowballC)
library(wordcloud)
library(dplyr)
getwd()
tweets =read.csv("out.csv", stringsAsFactors = F, row.names = 1,sep=",")
# A collection of text documents is called a Corpus
tweets_corpus = Corpus(VectorSource(tweets$tweet))
# View content of the first review
tweets_corpus[[1]]
tweets_corpus[1]
# Change to lower case, not necessary here
tweets_corpus = tm_map(tweets_corpus, content_transformer(tolower))
# Remove numbers
tweets_corpus = tm_map(tweets_corpus, removeNumbers)
# Remove punctuation marks and stopwords
#tweets_corpus = tm_map(tweets_corpus, removePunctuation)
tweets_corpus = tm_map(tweets_corpus, removeWords, c("duh", "whatever", stopwords("english")))
# Remove extra whitespaces
tweets_corpus =  tm_map(tweets_corpus, stripWhitespace)

tweets_corpus[[44]]$content


# Sometimes stemming is necessary
#tweets_corpus_stemmed =  tm_map(tweets_corpus, stemDocument)
#tweets_corpus_stemmed[[1]]$content


# Document-Term Matrix: documents as the rows, terms/words as the columns, frequency of the term in the document as the entries. Notice the dimension of the matrix
# You can use bounds in control to remove rare and frequent terms
tweets_dtm = DocumentTermMatrix(tweets_corpus, control = list(bounds = list(global = c(1, Inf))))
tweets_dtm
inspect(tweets_dtm[1:10, sample(ncol(tweets_dtm), 10)]) # 10 random columns

# Simple word cloud
findFreqTerms(tweets_dtm, 1000)
freq = data.frame(sort(colSums(as.matrix(tweets_dtm)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))

# Remove the less frequent terms such that the sparsity is less than 0.95
tweets_dtm = removeSparseTerms(tweets_dtm, 0.95)
tweets_dtm
# The first 10 documents
#inspect(tweets_dtm[1:10, sample(ncol(tweets_dtm), 10)]) # 10 random columns

# tf-idf(term frequency-inverse document frequency) instead of the frequencies of the term as entries, tf-idf measures the relative importance of a word to a document
#review_dtm_tfidf <- DocumentTermMatrix(review_corpus, control = list(weighting = weightTfIdf))
#review_dtm_tfidf = removeSparseTerms(review_dtm_tfidf, 0.95)
#review_dtm_tfidf
# The first 10 document
#inspect(review_dtm_tfidf[1:10,1:20])


# Precompiled list of words with positive and negative meanings
# Source: http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
neg_words = read.table("negative-words.txt", header = F, stringsAsFactors = F)[, 1]
pos_words = read.table("positive-words.txt", header = F, stringsAsFactors = F)[, 1]

tweets$neg = tm_term_score(tweets_dtm, neg_words)
tweets$pos = tm_term_score(tweets_dtm, pos_words)
#install.packages('xlsx')
library(xlsx)

test <- tweets$pos - tweets$neg
tweets$polarity <- as.numeric(test >= 0)

write.xlsx(tweets, "C:/Users/Keval Jain/Desktop/bia 658/tweets.xlsx")

# remove the actual texual content for statistical models
reviews$content = NULL
# construct the dataset for models
reviews = cbind(reviews, as.matrix(review_dtm))
reviews$polarity = as.factor(reviews$polarity)

# Split to testing and training set
id_train = sample(nrow(reviews),nrow(reviews)*0.80)
reviews.train = reviews[id_train,]
reviews.test = reviews[-id_train,]

# Compare 3 classification models
# classification tree, logistic regression, support vector machine (SVM)
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
install.packages("e1071")
library(e1071) # for Support Vector Machine

reviews.tree = rpart(polarity~.,  method = "class", data = reviews.train);
prp(reviews.tree)
reviews.glm = glm(polarity~ ., family = "binomial", data =reviews.train, maxit = 100);  
reviews.svm = svm(polarity~., data = reviews.train);

# Performance in the test set
pred.tree = predict(reviews.tree, reviews.test,  type="class")
table(reviews.test$polarity,pred.tree,dnn=c("Observed","Predicted"))
mean(ifelse(reviews.test$polarity != pred.tree, 1, 0))

pred.glm = as.numeric(predict(reviews.glm, reviews.test, type="response") > 0.5)
table(reviews.test$polarity,pred.glm,dnn=c("Observed","Predicted"))
mean(ifelse(reviews.test$polarity != pred.glm, 1, 0))

pred.svm = predict(reviews.svm, reviews.test)
table(reviews.test$polarity,pred.svm,dnn=c("Observed","Predicted"))
mean(ifelse(reviews.test$polarity != pred.svm, 1, 0))

