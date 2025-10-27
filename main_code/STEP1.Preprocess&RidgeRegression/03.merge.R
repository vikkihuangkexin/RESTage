organ<-"skin"
inter_threshold <- 0.15
setwd(paste0("data_demo/senescence/",organ,"/cell_interaction/sample/"))
dir.create(paste0("data_demo/senescence/",organ,"/cell_interaction/max_min/"))
files<-list.files(pattern="txt")
for ( i in 1:length(files)){
  a<-read.table(files[i],sep="\t",header=T,row.names=1,check.names=FALSE)

###max-min normalization
    b4<-(a-min(a))/(max(a)-min(a))
    write.table(b4,paste0("data_demo/senescence/",organ,"/cell_interaction/max_min/",files[i]),sep="\t",quote=F)
}

setwd(paste0("data_demo/senescence/",organ,"/cell_interaction/"))
for ( i in 1:length(files)){
  a<-read.table(paste0("max_min/",files[i]),sep="\t",header=T,row.names=1,check.names = FALSE)
  b<-a[,apply(a,2,max) > inter_threshold]
  if (is.data.frame(b)){
    c<-as.data.frame(colnames(b))
    colnames(c)<-"result"
    if (i == 1){
      result<-c
    }
    if (i != 1){
    result<-rbind(result,c)
    }
  }
}

result<-t(result)
colnames(result)<-result[1,]
write.table(result,paste0("data_demo/senescence/",organ,"/cell_interaction/max_min_merge_",
                          inter_threshold*100,".txt"),sep="\t",quote=F)