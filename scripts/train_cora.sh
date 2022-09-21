GENOTYPE=$1

python train.py \
--task 'node_level' \
--data 'Cora' \
--nb_classes 7 \
--in_dim_V 1433 \
--pos_encode 0 \
--batch 64 \
--node_dim 256 \
--dropout 0.2 \
--batchnorm_op \
--epochs 200 \
--lr 1e-3 \
--weight_decay 0.0 \
--optimizer 'ADAM' \
--load_genotypes $GENOTYPE \
--log_name 'node_cora_train.log'