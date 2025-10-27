library(Seurat)
library(SeuratDisk)
library("sceasy")
library("loomR")
library(stringr)
library(dplyr)
library(tidyr)
library(org.Hs.eg.db)

organ <- 'skin'
inter_threshold <- '15'

cylist <- read.table('data_demo/cytokine_list.txt',sep=',',header=T)
setwd(paste0('data_demo/data_impute'))
seu_obj <- readRDS('seurat.rds')

##### Labeling Data
genelist <- rownames(seu_obj@assays$RNA@counts)
sasp_activate <- read.table(paste0('data_demo/senescence/',organ,'/no_filtered/max_min_index_',inter_threshold,'/cytosig_activity.txt'),sep = '')
cell_activate <- read.table(paste0('data_demo/senescence/',organ,'/no_filtered/max_min_index_',inter_threshold,'/geneset_signature.txt'),sep='\t',header=T)
setwd(paste0('data_demo/senescence/',organ,'/no_filtered/max_min_index_',inter_threshold))
sasp_activate <- t(sasp_activate)
colnames(sasp_activate) <- sasp_activate[1,]
sasp_activate <- sasp_activate[-1,]
rownames(cell_activate) <- cell_activate[,1]
rownames(cell_activate) <- gsub("-", ".", rownames(cell_activate))
cl <- colnames(sasp_activate)
setwd(paste0('data_demo/senescence/',organ,'/no_filtered/max_min_index_',inter_threshold))
cor_data_df <- data.frame(cl)
for (i in 1:length(cl)) {
  x <- as.numeric(sasp_activate[,cl[i]])
  y <- as.numeric(cell_activate[,cl[i]])
  test <- cor.test(x,y,type="pearson")
  cor_data_df[i,2] <- test$estimate
  cor_data_df[i,3] <- test$p.value
  names(cor_data_df) <- c("cytokine","correlation","pvalue")
  }
cor_data_neg = cor_data_df %>% filter(pvalue < 0.01, correlation < 0)
write.table(cor_data_df,paste0('saspActivation$cellActivation.txt'),sep=",",quote = F)
write.table(cor_data_neg,paste0('saspActivation$cellActivation_P0.01CorrLowThan0.txt'),sep=",",quote = F)

# Generate positive sample genes
cl = intersect(rownames(sasp_activate), genelist)
cl = intersect(cl,cylist$cytokine)
cylist_P = read.table(paste0('data_demo/senescence/',organ,'/no_filtered/max_min_index_',inter_threshold,'/saspActivation$cellActivation_P0.01CorrLowThan0.txt'),sep = ',',header=T)

cl1 = cylist_P$cytokine
cl = intersect(cl,cl1)
cy <- data.frame(cytokine=cl)
write.table(cy,'WithCytokine.csv',,sep=',',quote=F,row.names=F)
merged_cy <- data.frame()
for (i in 1:length(cl)) {
  filePath = str_c("data_demo/cytokine/",cl[i],".csv", sep = '')
  current_cy = read.table(filePath,sep=',',header=T)
  merged_cy <- rbind(merged_cy, current_cy)
}
merged_cys <- merged_cy[,c(1)]
merged_cy <- data.frame(SYMBOL=c(cl,merged_cys))
merged_cy <- unique(merged_cy)
merged_cy <- merged_cy[which(merged_cy$SYMBOL %in% genelist),]
library(org.Hs.eg.db)
cols<-c("GENENAME","ENSEMBL")
pos_gene = AnnotationDbi::select(org.Hs.eg.db,keys = merged_cy,columns = cols,keytype = "SYMBOL")
pos_gene = pos_gene[!duplicated(pos_gene[,'SYMBOL']),]

merged_cy=pos_gene[,c('SYMBOL','ENSEMBL'),drop=FALSE]
merged_cy <- merged_cy[!is.na(merged_cy[, 'ENSEMBL']), ]

# Obtaining CCC cell sets
CCC = read.table(paste0('data_demo/senescence/',organ,'/cell_interaction/max_min_merge_',inter_threshold,'.txt'),header=T)
CCC = colnames(CCC)
dir.create(paste0('data_demo/senescence/',organ,'/Training'))

meta_data <- seu_obj@meta.data
meta_data$new_barcode<-gsub("-", ".", rownames(meta_data))
seu_obj<-RenameCells(seu_obj, new.names = meta_data$new_barcode)
seu_obj_ccc <- subset(seu_obj,cells=CCC)
#Calculate correlation and select negative sample genes
exprSet = seu_obj_ccc@assays[['RNA']]@data
exprSet = as.data.frame(t(as.matrix(exprSet)))
result <- setdiff(merged_cy$SYMBOL, cl)
exprSet <- exprSet[, -which(colnames(exprSet) %in% result)]
cor_data_neg_full <- data.frame()
lownum=0
for (i in 1:length(cl)) {
y <- as.numeric(exprSet[,cl[i]])
colnames <- colnames(exprSet)
cor_data_df <- data.frame(colnames)
cor_data_neg <- data.frame(colnames)
for (ii in 1:length(colnames)){
  test <- cor.test(as.numeric(exprSet[,ii]),y,method="pearson",alternative = "two.side",conf.level = 0.95)
  cor_data_df[ii,2] <- test$estimate
  cor_data_df[ii,3] <- test$p.value
}
names(cor_data_df) <- c("symbol","correlation","pvalue")
cor_data_neg = cor_data_df %>% filter(pvalue>0.01)
if (length(rownames(cor_data_neg))<=2000){
  lownum = lownum+1
    print(cl[i])
    print(lownum)
    next
}
if(length(colnames(cor_data_neg_full))==0){
  cor_data_neg_full=cor_data_neg
  print('reset')
  next
}
  cor_data_neg_f = merge(cor_data_neg_full, cor_data_neg, by = "symbol", all = FALSE)
  if (length(rownames(cor_data_neg_f))<=800){
    lownum = lownum+1
    print(cl[i])
    print(lownum)
  } else{
    cor_data_neg_full = cor_data_neg_f
  }
  print(paste0(length(rownames(cor_data_neg_full)),'  ',cl[i]))
}
corr_columns <- grep("^correlation", colnames(cor_data_neg_full), value=TRUE)
corr_mean <- rowMeans(cor_data_neg_full[, corr_columns])
corr_mean = abs(corr_mean)
p_columns <- grep("^pvalue", colnames(cor_data_neg_full), value=TRUE)
p_mean <- rowMeans(cor_data_neg_full[, p_columns])
neg_result=cor_data_neg_full[,c('symbol'),drop=FALSE]
neg_result[,2]=corr_mean
neg_result[,3]=p_mean
names(neg_result) <- c("SYMBOL","correlation","pvalue")

cols<-c("GENENAME","ENSEMBL")
neg_gene = AnnotationDbi::select(org.Hs.eg.db,keys = neg_result$SYMBOL,columns = cols,keytype = "SYMBOL")
neg_gene = neg_gene[!duplicated(neg_gene[,'SYMBOL']),]
neg_gene = neg_gene[,c('ENSEMBL')]
neg_result$ENSEMBL=neg_gene
neg_result <- neg_result[!is.na(neg_result[, 'ENSEMBL']), ]
sorted_table <- neg_result[order(neg_result[, "correlation"], decreasing=TRUE), ]
######if length(gene) too short
if (length(rownames(sorted_table))>=length(rownames(merged_cy))){
  gene <- sorted_table[1:length(rownames(merged_cy)),c("SYMBOL",'ENSEMBL'),drop=FALSE ]
}else{
  gene <- sorted_table[,c("SYMBOL",'ENSEMBL'),drop=FALSE ]
}
write.table(gene,paste0('data_demo/senescence/',organ,'/Training/neg_P.csv'),sep=',',quote=F,row.names=F)
write.table(merged_cy,paste0('data_demo/senescence/',organ,'/Training/pos_P.csv'),sep=',',quote=F,row.names=F)
metadata_ccc = seu_obj_ccc@meta.data
group <- table(metadata_ccc$group)
group <- as.data.frame(group)
ID <- table(metadata_ccc$ID)
ID <- as.data.frame(ID)
cell_type<- table(metadata_ccc$ident)
cell_type<- as.data.frame(cell_type)
infor = rbind(group,cell_type,ID)
infor1  = data.frame(Var1 = character(4))
infor1$Var1 = c('sample','cy_num','pos_gene','neg_gene')
infor1$Freq = c(length(CCC),length(cl),length(rownames(merged_cy)),length(rownames(gene)))
infor = rbind(infor1,infor)
infor <- t(infor)
colnames(infor) <- infor[1, ]
write.table(infor,paste0('data_demo/senescence/',organ,'/data_information.csv'),sep = ',',quote=F,row.names=F)


#### Use limma to find differentially expressed genes
table(seu_obj_ccc@meta.data$ident)
seu_obj_T = subset(seu_obj_ccc,subset=ident==CT)
data <- GetAssayData(seu_obj_T, assay ='RNA', slot = 'counts')
cell_metadata <- seu_obj_T@meta.data
gene_annotation <-data.frame(gene_short_name = rownames(data))
rownames(gene_annotation) <-rownames(data)
posgene = setdiff(merged_cy1$SYMBOL, cy$cytokine)
library(limma)
library(edgeR)
design <- model.matrix(~0+factor(cell_metadata$group))
colnames(design) <- levels(factor(cell_metadata$group))
rownames(design) <- colnames(data)

dge <- DGEList(counts=data)
dge <- calcNormFactors(dge)
cont.matrix <- makeContrasts(contrasts=paste0('old-mid'),
                             levels = design)
de <- voom(dge,design,plot=TRUE, normalize="quantile")
fit1 <- lmFit(de, design)
fit2 <- contrasts.fit(fit1,cont.matrix)
efit <- eBayes(fit2, trend=F)
tempDEG <- topTable(efit, coef=paste0('old-mid'), n=Inf)
DEG_limma_voom  <- na.omit(tempDEG)
DEG_limma_voom = DEG_limma_voom[rownames(DEG_limma_voom) %in% posgene,]
write.table(DEG_limma_voom,file=paste0(cell_type,"-all_marker_MakerWithFullGene_overlapInteraction.csv"),sep=",",quote = F)


#### Saving loom
exprSet = seu_obj_ccc@assays[['RNA']]@data
exprSet = as.data.frame(as.matrix(exprSet))
#Replace expression value with cytosig_activity
sasp_activate = read.table(paste0('data_demo/senescence/',organ,'/no_filtered/max_min_index_',inter_threshold,'/cytosig_activity.txt'),sep = '')

for (cy in rownames(sasp_activate)){
  if(cy %in% cl)
  exprSet[cy,]=sasp_activate[cy,]
}
metadata = seu_obj_ccc@meta.data
counts <- Matrix::Matrix(as.matrix(exprSet), sparse = T)
options(Seurat.object.assay.version = "v3")
seu_obj_ccc1 <- CreateSeuratObject(counts,meta.data = metadata)
saveRDS(seu_obj_ccc1,'seurat_cccFullGene_P.rds')

seu_obj.loom <- as.loom(seu_obj_ccc1, filename = "seurat.impute.loom", verbose = FALSE)
seu_obj.loom <- connect(filename = "seurat.impute.loom", mode = "r+",skip.validate=TRUE)
seu_obj.loom

gene_name =rownames(seu_obj_ccc1)
seu_obj.loom[['row_attrs/gene_name']] = gene_name
cols<-c("GENENAME","ENSEMBL")
ensembl_id = AnnotationDbi::select(org.Hs.eg.db,keys = gene_name,columns = cols,keytype = "SYMBOL")
ensembl_id = ensembl_id[!duplicated(ensembl_id),]
ensembl_id = ensembl_id[!duplicated(ensembl_id[,'SYMBOL']),]
for (i in length(rownames(merged_cy))){
  ensembl_id[which(ensembl_id[,'SYMBOL']==merged_cy[i,'SYMBOL']),'ENSEMBL'] = merged_cy[i,'ENSEMBL']
}
for (i in length(rownames(gene))){
  ensembl_id[which(ensembl_id[,'SYMBOL']==gene[i,'symbol']),'ENSEMBL'] = gene[i,'ENSEMBL']
}
ensembl.id = ensembl_id$ENSEMBL
seu_obj.loom[['col_attrs/n_counts']] = seu_obj_ccc1@meta.data$nCount_RNA
seu_obj.loom[['row_attrs/ensembl_id']] = ensembl.id
seu_obj.loom[['row_attrs']]
seu_obj.loom[['col_attrs']]
seu_obj.loom$close_all()
