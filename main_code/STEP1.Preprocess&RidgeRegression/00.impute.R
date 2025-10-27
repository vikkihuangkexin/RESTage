rm(list = ls())
library(Seurat)
library(SAVER)
library(stringr)
library(Matrix)
library(dplyr)

setwd('data_demo')
dir.create('data_impute')
#####Load data####
seurat <- readRDS('data_impute/skin_ori.rds')
organ='skin'

###Quality Control
dir.create('QC')
processed[["percent.mt"]] <- PercentageFeatureSet(processed,
                                                             pattern = "^MT-")
pdf(paste0("QC/QC-VlnPlot.pdf"),width = 8,height = 4.5)
VlnPlot(processed, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"),
        ncol = 3)
dev.off()

##QC: Statistics of gene count, RNA, and mitochondrial gene distribution
gene.freq <- do.call("cbind", tapply(processed@meta.data$nFeature_RNA,processed@meta.data$orig.ident,quantile,probs=seq(0,1,0.05)))
rna.freq <- do.call("cbind", tapply(processed@meta.data$nCount_RNA,processed@meta.data$orig.ident,quantile,probs=seq(0,1,0.05)))
mt.freq <- do.call("cbind", tapply(processed@meta.data$percent.mt,processed@meta.data$orig.ident,quantile,probs=seq(0,1,0.05)))
freq.combine <- as.data.frame(cbind(gene.freq,rna.freq,mt.freq))
colnames(freq.combine) <- c("Count_Gene","Count_RNA","MT_percent")
View(freq.combine)
write.table(freq.combine,file = paste0("QC/QC-gene_frequency.txt"),quote = F,sep = "\t")
rm(gene.freq,rna.freq,mt.freq)


##QC: Correlation between gene number and distribution of mitochondrial genes and RNA quantity
plot1 <- FeatureScatter(processed, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(processed, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
pdf(paste0("QC/QC-FeatureScatter.pdf"),width = 8,height = 4.5)
CombinePlots(plots = list(plot1, plot2),legend = "none")
dev.off()
rm(plot1,plot2)

#### QC: Filter Cell ####
cat("Before filter :",nrow(processed@meta.data),"cells\n")
#Filtering cells primarily considers cell quality.
# The proportion of mitochondrial genes should be kept below 10%.
# The number of expressed genes should be at least 200.
# The number of detected RNA molecules should not be less than 1,000 (too few indicates insufficient data from the cell)
# nor greater than 20,000 (too many indicates impure cells).
processed <- subset(processed,
                            subset =
                              nFeature_RNA > 200 &
                              nCount_RNA > 1000 &
                              nCount_RNA < 20000 &
                              percent.mt < 10)
cat("After filter :",nrow(processed@meta.data),"cells\n")
saveRDS(processed,'seurat_QC.rds')

cell_exp = as.matrix(processed@assays$RNA@counts)
####Run Saver####
cell_exp.saver <- saver(cell_exp, ncores = 4,estimates.only = TRUE,size.factor = 1)
str(cell_exp.saver)
####large database####
# saver1 <- saver(cell_exp, pred.genes = 1:10000, pred.genes.only = TRUE,
#                 do.fast = FALSE,ncores = 4)
# saver2 <- saver(cell_exp, pred.genes = 10001:20000, pred.genes.only = TRUE,
#                 do.fast = FALSE,ncores = 4)
# saver3 <- saver(cell_exp, pred.genes = 20002:30000, pred.genes.only = TRUE,
#                 do.fast = FALSE,ncores = 8)
# saver4 <- saver(cell_exp, pred.genes = 30000:length(rownames(cell_exp)), pred.genes.only = TRUE,
#                 do.fast = FALSE,ncores = 8)
# cell_exp.saver <- combine.saver(list(saver1, saver2, saver3, saver4))
####Save count martix only####
write.table(cell_exp.saver,"data_impute/impute.txt",sep="\t",quote=F)
###Save as seurat object####
seurat.impute <- CreateSeuratObject(cell_exp.saver,meta.data=processed@meta.data)
saveRDS(seurat.impute,"data_impute/seurat.impute.rds")