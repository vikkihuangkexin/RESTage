
library(readxl)
library(Seurat)
library(GSEABase)
library(escape)
library(AUCell)
library(tidyverse)
library(dittoSeq)
library(gridExtra)
library(patchwork)
library(MAST)
library(openxlsx)
library(reshape2)
library(ggpubr)

# Set working directory and read Seurat object
setwd("data_demo/data_impute/")
seu_obj <- readRDS("data_demo/data_impute/seurat.impute.rds")
seu_obj@meta.data$seurat_clusters <- as.factor(seu_obj@meta.data$seurat_clusters)

# Create age group metadata: <= 60 -> "mid", > 60 -> "old"
seu_obj@meta.data$group[which(seu_obj@meta.data$age <= 60)] <- "mid"
seu_obj@meta.data$group[which(seu_obj@meta.data$age > 60)] <- "old"

# Define gene list and convert to GeneSetCollection
X3datasets <- c(
  "ACVR1B", "ANG","ANGPT1","ANGPTL4","AREG","AXL","BEX3","BMP2","BMP6","C3",
  "CCL1","CCL13","CCL16","CCL2","CCL20","CCL24","CCL26","CCL3","CCL3L1","CCL4",
  "CCL5","CCL7","CCL8","CD55","CD9","CSF1","CSF2","CSF2RB","CST4","CTNNB1",
  "CTSB","CXCL1","CXCL10","CXCL12","CXCL16","CXCL2","CXCL3","CXCL8","CXCR2",
  "DKK1","EDN1","EGF","EGFR","EREG","ESM1","ETS2","FAS","FGF1","FGF2","FGF7",
  "GDF15","GEM","GMFG","HGF","HMGB1","ICAM1","ICAM3","IGF1","IGFBP1","IGFBP2",
  "IGFBP3","IGFBP4","IGFBP5","IGFBP6","IGFBP7","IL10","IL13","IL15","IL18",
  "IL1A","IL1B","IL2","IL32","IL6","IL6ST","IL7","INHA","IQGAP2","ITGA2","ITPKA",
  "JUN","KITLG","LCP1","MIF","MMP1","MMP10","MMP12","MMP13","MMP14","MMP2","MMP3",
  "MMP9","NAP1L4","NRG1","PAPPA","PECAM1","PGF","PIGF","PLAT","PLAU","PLAUR",
  "PTBP1","PTGER2","PTGES","RPS6KA5","SCAMP4","SELPLG","SEMA3F","SERPINB4",
  "SERPINE1","SERPINE2","SPP1","SPX","TIMP2","TNF","TNFRSF10C","TNFRSF11B",
  "TNFRSF1A","TNFRSF1B","TUBGCP2","VEGFA","VEGFC","VGF","WNT16","WNT2"
)
X3datasets <- as.data.frame(X3datasets)
colnames(X3datasets)[1] <- "Gene"

SEN_Mayo <- X3datasets$Gene
gs1  <- GeneSet(SEN_Mayo)
geneSets <- list(geneSet1 = gs1)
geneSets$geneSet1@setName <- "SEN_Mayo"
geneSets <- GeneSetCollection(geneSets)

# Calculate Enrichment Score (ES). Adjust cores as appropriate for environment
ES <- enrichIt(obj = seu_obj, gene.sets = geneSets, groups = 1000, cores = 8)

# Add gene set score to Seurat metadata and classify top 10% as "SASP cells"
seu_obj <- AddMetaData(seu_obj, ES)
q90 <- quantile(seu_obj@meta.data$SEN_Mayo, 0.90)
seu_obj@meta.data <- transform(seu_obj@meta.data,
                               cluster_id_2 = ifelse(seu_obj@meta.data$SEN_Mayo > q90, "SASP cells", "Other"))
table(seu_obj@meta.data$cluster_id_2)
head(seu_obj@meta.data)

saveRDS(seu_obj, "seurat.rds")

# Plot UMAPs
pdf("umap.pdf", width = 7, height = 6)
DimPlot(seu_obj, reduction = "umap", group.by = "cluster_id_2",
        cols = c("grey", "#CE1261"), order = "SASP cells", pt.size = 0.5,
        label = TRUE, repel = TRUE)
dev.off()

pdf("celltype_umap.pdf", width = 7, height = 6)
DimPlot(object = seu_obj, reduction = "umap", group.by = "ident")
dev.off()

# Violin plot example for CDKN1A
gene_sig <- c("CDKN1A")
comparisons <- list(c("SASP cells", "Other"))
vp_case1(gene_signature = gene_sig, file_name = "CDKN_/CDKN1A", test_sign = comparisons, y_max = 7)

VlnPlot(seu_obj, features = "CDKN1A", pt.size = 0, group.by = "cluster_id_2") +
  NoLegend() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  geom_signif(comparisons = list(c("SASP cells", "Other")),
              map_signif_level = function(p) sprintf("*p = %.2g", p),
              step_increase = 0.15) +
  ylim(-2, 5)

# Create directory and save several violin plots
dir.create("CDKN_", showWarnings = FALSE)
pdf("CDKN_/CDKN1A.pdf", width = 7, height = 6)
VlnPlot(seu_obj, group.by = "cluster_id_2", "CDKN1A", pt.size = 0, log = TRUE, y.max = 7) +
  theme(axis.title.x = element_blank()) + ylab("Log UMI") & NoLegend()
dev.off()

pdf("CDKN_/CDKN1B.pdf", width = 7, height = 6)
VlnPlot(seu_obj, group.by = "cluster_id_2", "CDKN1B", pt.size = 0, log = TRUE) +
  theme(axis.title.x = element_blank()) + ylab("Log UMI") & NoLegend()
dev.off()

pdf("CDKN_/CDKN2A.pdf", width = 7, height = 6)
VlnPlot(seu_obj, group.by = "cluster_id_2", "CDKN2A", pt.size = 0, log = TRUE) +
  theme(axis.title.x = element_blank()) + ylab("Log UMI") & NoLegend()
dev.off()

pdf("CDKN_/CDKN2B.pdf", width = 7, height = 6)
VlnPlot(seu_obj, group.by = "cluster_id_2", "CDKN2B", pt.size = 0, log = TRUE) +
  theme(axis.title.x = element_blank()) + ylab("Log UMI") & NoLegend()
dev.off()

pdf("CDKN_/TGFB1.pdf", width = 7, height = 6)
VlnPlot(seu_obj, group.by = "cluster_id_2", "TGFB1", pt.size = 0, log = TRUE) +
  theme(axis.title.x = element_blank()) + ylab("Log UMI") & NoLegend()
dev.off()

# Cell cycle scoring and phase counts
{
  s.genes <- Seurat::cc.genes.updated.2019$s.genes
  g2m.genes <- Seurat::cc.genes.updated.2019$g2m.genes
  seu_obj <- CellCycleScoring(seu_obj, s.features = s.genes, g2m.features = g2m.genes, set.ident = TRUE)
  SASP_seu <- subset(seu_obj, cluster_id_2 == "SASP cells")
  other_seu <- subset(seu_obj, cluster_id_2 == "Other")
  write.xlsx(table(other_seu@meta.data$Phase), "phase_all.xlsx")
  write.xlsx(table(SASP_seu@meta.data$Phase), "phase_justSASP.xlsx")

  dt_all = data.frame(A = c(table(other_seu@meta.data$Phase)[1],
                             table(other_seu@meta.data$Phase)[2],
                             table(other_seu@meta.data$Phase)[3]),
                      B = c("G1","G2M","S"))
  dt_sasp = data.frame(A = c(table(SASP_seu@meta.data$Phase)[1],
                              table(SASP_seu@meta.data$Phase)[2],
                              table(SASP_seu@meta.data$Phase)[3]),
                       B = c("G1","G2M","S"))

  myLabel_all = paste(as.vector(dt_all$B), "(", round(dt_all$A / sum(dt_all$A) * 100, 2), "%)", sep = "")
  myLabel_sasp = paste(as.vector(dt_sasp$B), "(", round(dt_sasp$A / sum(dt_sasp$A) * 100, 2), "%)", sep = "")

  perct_all <- paste(round(dt_all$A / sum(dt_all$A) * 100, 2), "%")
  dt_all_res <- dt_all
  dt_all_res$Percentage <- perct_all
  colnames(dt_all_res) <- c("Celltype", "Num", "Percentage")

  perct_sasp <- paste(round(dt_sasp$A / sum(dt_sasp$A) * 100, 2), "%")
  dt_sasp_res <- dt_sasp
  dt_sasp_res$Percentage <- perct_sasp
  colnames(dt_sasp_res) <- c("Celltype", "Num", "Percentage")

  write.csv(dt_all_res, 'cell_proportion_all.csv', row.names = FALSE)
  write.csv(dt_sasp_res, 'cell_proportion_SASP.csv', row.names = FALSE)

  p2 = ggplot(dt_all, aes(x = "", y = A, fill = B)) +
    geom_bar(stat = "identity", width = 0.5) +
    labs(x = "", y = "", title = "cell_cycle_other") +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.ticks = element_blank()) +
    theme(legend.title = element_blank(), legend.position = "right") +
    scale_fill_discrete(breaks = dt_all$B, labels = myLabel_all) +
    theme(panel.grid = element_blank()) +
    theme(axis.text.x = element_blank())

  p3 = ggplot(dt_sasp, aes(x = "", y = A, fill = B)) +
    geom_bar(stat = "identity", width = 0.5) +
    labs(x = "", y = "", title = "cell_cycle_sasp") +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.ticks = element_blank()) +
    theme(legend.title = element_blank(), legend.position = "right") +
    scale_fill_discrete(breaks = dt_sasp$B, labels = myLabel_sasp) +
    theme(panel.grid = element_blank()) +
    theme(axis.text.x = element_blank())

  library(patchwork)
  p2 + p3
  ggsave("cell_cycle.pdf", width = 10, height = 5)
}

# Chi-squared test comparing phase distribution
M <- as.table(rbind(
  c(table(seu_obj@meta.data$Phase)[1], table(seu_obj@meta.data$Phase)[2], table(seu_obj@meta.data$Phase)[3]),
  c(table(SASP_seu@meta.data$Phase)[1], table(SASP_seu@meta.data$Phase)[2], table(SASP_seu@meta.data$Phase)[3])
))
dimnames(M) <- list(ident = c("All", "SASP"), cycle = c("G1", "G2M", "S"))
chisq.test(M)

# Proportion of SASP and Other across sample groups (age groups)
{
  SASP_seu <- subset(seu_obj, cluster_id_2 == "SASP cells")
  other_seu <- subset(seu_obj, cluster_id_2 == "Other")

  # Build data frames depending on number of groups present
  if (length(table(SASP_seu@meta.data$group)) == 3) {
    dt_SASP = data.frame(SASP = as.numeric(table(SASP_seu@meta.data$group)),
                         ident = names(table(SASP_seu@meta.data$group)))
    dt_other = data.frame(Other = as.numeric(table(other_seu@meta.data$group)),
                          ident = names(table(other_seu@meta.data$group)))
  } else {
    dt_SASP = data.frame(SASP = as.numeric(table(SASP_seu@meta.data$group)),
                         ident = names(table(SASP_seu@meta.data$group)))
    dt_other = data.frame(Other = as.numeric(table(other_seu@meta.data$group)),
                          ident = names(table(other_seu@meta.data$group)))
  }

  df = merge(dt_other, dt_SASP, by = "ident")
  if (length(levels(factor(SASP_seu@meta.data$group))) == 3) {
    df$ident <- factor(df$ident, levels = c("youth", "mid", "old"))
  }

  # Compute proportions of SASP and Other within each age group
  df$SASP_Prop <- df$SASP / rowSums(df[, c("SASP", "Other")])
  df$Other_Prop <- df$Other / rowSums(df[, c("SASP", "Other")])

  # Convert to long format for plotting
  df_long <- tidyr::pivot_longer(df, c("SASP_Prop", "Other_Prop"), names_to = "Category", values_to = "Proportion")

  # Plot bar charts with proportion labels
  p <- ggplot(df_long, aes(x = ident, y = Proportion, fill = Category)) +
    geom_bar(stat = "identity", color = "white", position = "dodge") +
    labs(x = NULL, y = "Proportion", fill = NULL, title = "Bar Charts of SASP and Other by Age Group") +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, size = 14),
          axis.title.x = element_blank(),
          legend.position = "right",
          legend.title = element_blank(),
          legend.text = element_text(size = 12),
          legend.key.width = unit(0.6, "cm"),
          legend.key.height = unit(0.6, "cm"))

  p + geom_text(aes(label = paste0(scales::percent(Proportion), " (", Category, ")")),
                position = position_dodge(width = 0.9), size = 3, vjust = -0.5)

  ggsave("cell_sample.pdf", width = 10, height = 5)
}

# Differential expression markers (SASP cells vs Other)
DefaultAssay(seu_obj) <- 'RNA'
dir.create("txt", showWarnings = FALSE)
dir.create("csv", showWarnings = FALSE)

different_express_gene <- FindMarkers(seu_obj, group.by = "cluster_id_2",
                                      ident.1 = "SASP cells",
                                      logfc.threshold = 0.2, test.use = "MAST",
                                      only.pos = TRUE, min.pct = 0.1)
write.table(different_express_gene, file = "txt/all_marker.txt", sep = "\t", quote = FALSE)
write.table(different_express_gene, file = "csv/all_marker.csv", sep = ",", quote = FALSE)

different_express_gene <- different_express_gene[different_express_gene$p_val_adj <= 0.05, ]
write.table(different_express_gene, file = "txt/marker.txt", sep = "\t", quote = FALSE)
write.table(different_express_gene, file = "csv/marker.csv", sep = ",", quote = FALSE)

# GO enrichment analysis using gprofiler2
library(gprofiler2)
gostres <- gost(query = rownames(different_express_gene),
                organism = "hsapiens", ordered_query = FALSE,
                multi_query = FALSE, significant = TRUE, exclude_iea = FALSE,
                measure_underrepresentation = FALSE, evcodes = FALSE,
                user_threshold = 0.05, correction_method = "fdr",
                domain_scope = "annotated", custom_bg = NULL,
                numeric_ns = "", as_short_link = FALSE,
                sources = c("GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "HP", "HPA", "WP"))
go_result <- cbind(gostres$result$term_id, gostres$result$term_name, gostres$result$p_value, gostres$result$significant)
write.table(go_result, "txt/go_result.txt", sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(go_result, "csv/go_result.csv", sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
p <- gostplot(gostres, capped = FALSE, interactive = FALSE)
pdf("TEST_go_plot.pdf", width = 10, height = 8)
pp <- publish_gostplot(p, highlight_terms = c("GO:0090402","GO:0090398","GO:0007049","GO:0051726","GO:0006977","GO:0036094", "GO:0005615", "GO:0030141", "GO:0099503", "GO:0007050", "GO:0008285", "0010941", "GO:0140352", "KEGG:04911", "KEGG:04218","REAC:R-HSA-382551", "WP:WP254"),
                       width = NA, height = NA, filename = NULL)
dev.off()

# Full enrichment results
gostres <- gost(query = rownames(different_express_gene),
                organism = "hsapiens", ordered_query = FALSE,
                multi_query = FALSE, significant = FALSE, exclude_iea = FALSE,
                measure_underrepresentation = FALSE, evcodes = FALSE,
                user_threshold = 1, correction_method = "fdr",
                domain_scope = "annotated", custom_bg = NULL,
                numeric_ns = "", as_short_link = FALSE,
                sources = c("GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "HP", "HPA", "WP"))
go_result <- cbind(gostres$result$term_id, gostres$result$term_name, gostres$result$p_value, gostres$result$significant, gostres$result$term_size)
write.table(go_result, "txt/all_go_result.txt", sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
write.table(go_result, "csv/all_go_result.csv", sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
p <- gostplot(gostres, capped = FALSE, interactive = FALSE)
pdf("all_go_plot.pdf", width = 10, height = 8)
pp <- publish_gostplot(p, highlight_terms = c("GO:0090402","GO:0090398","GO:0007049","GO:0051726","GO:0006977","GO:0036094", "GO:0005615", "GO:0030141", "GO:0099503", "GO:0007050", "GO:0008285", "0010941", "GO:0140352", "KEGG:04911", "KEGG:04218","REAC:R-HSA-382551", "WP:WP254"),
                       width = NA, height = NA, filename = NULL)
dev.off()

# Annotation and clusterProfiler enrichment
library(clusterProfiler)
library(org.Hs.eg.db)
library(enrichplot)
library(ggplot2)
library(stringi)  # string handling package
library(GOplot)

gene_list <- read.table("txt/marker.txt", sep = "\t", header = TRUE)
table(duplicated(rownames(gene_list)))

gene <- mapIds(org.Hs.eg.db, rownames(gene_list), 'ENTREZID', 'SYMBOL')
gene <- as.data.frame(gene)
gene <- gene[is.na(gene[, 1]) == FALSE, ]

# Enrich GO (all)
GO <- enrichGO(gene = gene,
               OrgDb = org.Hs.eg.db,
               pvalueCutoff = 1,
               qvalueCutoff = 1,
               pAdjustMethod = "fdr",
               minGSSize = 0,
               ont = "all",
               readable = TRUE)
write.table(GO, file = "txt/all_GO.txt", sep = "\t", quote = FALSE, row.names = FALSE)
write.table(GO, file = "csv/all_GO.csv", sep = ",", quote = FALSE, row.names = FALSE)

# Enrich GO (significant)
GO <- enrichGO(gene = gene,
               OrgDb = org.Hs.eg.db,
               pvalueCutoff = 0.05,
               qvalueCutoff = 0.1,
               pAdjustMethod = "fdr",
               ont = "all",
               readable = TRUE)
write.table(GO, file = "txt/sig_GO.txt", sep = "\t", quote = FALSE, row.names = FALSE)
write.table(GO, file = "csv/sig_GO.csv", sep = ",", quote = FALSE, row.names = FALSE)

ego <- enrichKEGG(gene = gene,
                  keyType = "kegg",
                  organism = 'hsa',
                  pvalueCutoff = 0.05,
                  pAdjustMethod = "fdr",
                  qvalueCutoff = 0.1)
write.table(ego, "txt/kegg.txt", sep = "\t", quote = FALSE)
write.table(ego, "csv/kegg.csv", sep = ",", quote = FALSE)

# Plot violin plots for specified gene sets if they exist in the results
sene_gene <- c("AKT3","MIR543","CDK2","CDK6","CDKN1A","ZMPSTE24","CDKN1B","CDKN2A","CDKN2B","CITED2","KAT5","PLK2","NEK6","ZNF277","CGAS","MAPK14","VASH1","PLA2R1","SMC5","SIRT1","MORC3","NUP62","ABL1","ULK3","RSL1D1","FBXO5","MAGEA2B","NSMCE2","H2AX","HLA-G","HMGA1","HRAS","ID2","IGF1R","ING2","KIR2DL4","ARG2","LMNA","ARNTL","MIR10A","MIR146A","MIR17","MIR188","MIR217","MIR22","MIR34A","MAGEA2","MAP3K3","MAP3K5","MIF","MNT","ATM","NPM1","YBX1","OPA1","PAWR","ABI3","FZR1","WNT16","SIRT6","PML","PRMT6","PRELP","SLC30A10","PRKCD","MAPK8","MAPK11","MAPK9","MAPK10","MAP2K1","MAP2K3","MAP2K6","MAP2K7","B2M","ZMIZ1","PTEN","MIR20B","RBL1","BCL6","MAP2K4","BMPR1A","SPI1","SRF","BRCA2","NEK4","TBX2","TBX3","MIR590","TERC","TERF2","TERT","TP53","TWIST1","WNT1","WRN","SMC6","KAT6A","ZKSCAN3","HMGA2","CALR","YPEL3","ECRG4","MAPKAPK5","TP63","PNPT1","DNAJA3","EEF1E1","NUAK1")
sene_gene <- sene_gene[sene_gene %in% rownames(different_express_gene)]
dir.create("senescence_gene", showWarnings = FALSE)
for (i in seq_along(sene_gene)) {
  VlnPlot(seu_obj, group.by = "cluster_id_2", sene_gene[i], pt.size = 0, log = TRUE) +
    theme(axis.title.x = element_blank()) + ylab("Log UMI") & NoLegend()
  ggsave(paste0("senescence_gene/", sene_gene[i], ".pdf"), width = 7, height = 6)
}

# Cell cycle arrest genes
arrest_gene <- c("ATM","CASP2","CDKN1A","CDKN1B","PLK3","GML","MDM2","MUC1","PML","RPL26","SOX4","TP53","CRADD","PLK2","CHEK2","TRIAP1","GTSE1","PIDD1","PRAP1")
arrest_gene <- arrest_gene[arrest_gene %in% rownames(different_express_gene)]
dir.create("arrest_gene", showWarnings = FALSE)
for (i in seq_along(arrest_gene)) {
  VlnPlot(seu_obj, group.by = "cluster_id_2", arrest_gene[i], pt.size = 0, log = TRUE) +
    theme(axis.title.x = element_blank()) + ylab("Log UMI") & NoLegend()
  ggsave(paste0("arrest_gene/", arrest_gene[i], ".pdf"), width = 7, height = 6)
}

# Cell age gene set
cellage_gene <- c("C9orf40","CDC25A","CDCA4","CKAP2","GTF3C4","HAUS4","IMMT","MCM7","MTHFD2","MYBL2","NEK2","NIPA2","TCEB3")
cellage_gene <- cellage_gene[cellage_gene %in% rownames(seu_obj)]
dir.create("cellage_geneset", showWarnings = FALSE)
for (i in seq_along(cellage_gene)) {
  VlnPlot(seu_obj, group.by = "cluster_id_2", cellage_gene[i], pt.size = 0, log = TRUE) +
    theme(axis.title.x = element_blank()) + ylab("Log UMI") & NoLegend()
  ggsave(paste0("cellage_geneset/", cellage_gene[i], ".pdf"), width = 7, height = 6)
}

# Export counts for SASP subset and for T cells (adjust paths as needed)
seu_obj <- readRDS("data_demo/data_impute/seurat.rds")
dir.create("skin", showWarnings = FALSE)
aaa <- subset(seu_obj, cluster_id_2 == "SASP cells")
write.table(as.matrix(aaa@assays$RNA@counts), "data_demo/senescence/skin/SASP_counts.txt", sep = "\t", quote = FALSE)
table(seu_obj@meta.data$ident)
bbb <- subset(seu_obj, cell_type == "T")
write.table(as.matrix(bbb@assays$RNA@counts), "data_demo/senescence/skin/T_cell_counts.txt", sep = "\t", quote = FALSE)
