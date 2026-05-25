# Sweep 3: Pass@k (pass_at_k_open)
# Split: stratified
# Cross-k predictions enabled (default)

method_collections=(
    continuous_mrmr
    continuous_irt
    other_baslines
    anchor_points
)

datasets=(pass_at_k_open_datasets)

# coreset_sizes=(25 50 75 100 5% 10% 15%)
coreset_sizes=(5% 10% 15%)
nmodel_trains=(10 15 20)

num_run=5

for dataset in ${datasets[@]}; do
    for methods in ${method_collections[@]}; do
        for nmodel_train in ${nmodel_trains[@]}; do
            for coreset_size in ${coreset_sizes[@]}; do
                python main.py --coreset_size $coreset_size --model_split_method stratified --methods $methods --datasets $dataset --num_train_models $nmodel_train --num_run $num_run
            done
        done
    done
done
