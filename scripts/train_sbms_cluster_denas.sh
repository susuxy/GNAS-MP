GENOTYPE=$1

python train_random_search.py \
--task 'node_level' \
--data 'SBM_CLUSTER' \
--nb_classes 6 \
--in_dim_V 7 \
--pos_encode 0 \
--batch 64 \
--node_dim 70 \
--dropout 0.2 \
--batchnorm_op \
--epochs 200 \
--lr 1e-3 \
--weight_decay 0.0 \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE \
--log_name 'node_cluster_train_denas.log'