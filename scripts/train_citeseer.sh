#GENOTYPE="archs/folder5/Citeseer/0.yaml"
GENOTYPE="archs/folder5/CoauthorPhysics/0.yaml"

python train.py \
--task 'node_level' \
--data 'Citeseer' \
--nb_classes 6 \
--in_dim_V 3703 \
--pos_encode 0 \
--batch 1 \
--node_dim 32 \
--dropout 0.5 \
--epochs 1 \
--lr 1e-4 \
--weight_decay 0 \
--optimizer 'ADAM' \
--batchnorm_op \
--load_genotypes $GENOTYPE \
--log_name 'node_citeseer_train.log'