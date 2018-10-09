
set.seed(100)
library(readr)
library(cluster) 
library(fpc)
library(dplyr)

df <- read_csv("beatsdataset.csv")

df <- df %>%
  select(-c(X1, class)) %>%
  slice(seq(1, 2300, 10)) #subset data so that every 10th observation is selected

#wss <- (nrow(df) - 1) * sum(apply(df, 2, var)) #calculate within groups sum of squares to select number of clusters
#for (i in 2:15) wss[i] <- sum(kmeans(df, centers=i)$withinss)

fit <- kmeans(df, 4)

#aggregate(df,by=list(fit$cluster),FUN=mean) #summary statistics for each cluster

cluster <- data.frame(df, fit$cluster)

#clusplot(df, fit$cluster, color=TRUE, shade=TRUE, labels=2, lines=0) #cluster plot
#plotcluster(df, fit$cluster)

discoords <- discrcoord(df, fit$cluster)
#plot(discoords$proj, col = fit$cluster)

cluster <- data.frame(cluster, discoords$proj[,1:2])
# plot(mydata$X1, mydata$X2, col = mydata$fit.cluster)

#write.csv(cluster, "clusterdf.csv")
