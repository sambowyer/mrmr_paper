# Sweep 1: Binary (openllm + helm)
# Split: binned_interpolation (same as stratified, but doesn't need to consider model temperatures)

method_collections=(
    binary_mrmr
    binary_irt
    other_baslines
    anchor_points
)

datasets=(openllm_datasets helm_datasets)

# coreset_sizes=(50 100 250 5% 10% 15%)
coreset_sizes=(5% 10% 15%)
nmodel_trains=(15 30 50)

num_run=5

for dataset in ${datasets[@]}; do
    for methods in ${method_collections[@]}; do
        for nmodel_train in ${nmodel_trains[@]}; do
            for coreset_size in ${coreset_sizes[@]}; do
                python main.py --coreset_size $coreset_size --model_split_method binned_interpolation --methods $methods --datasets $dataset --num_train_models $nmodel_train --num_run $num_run
            done
        done
    done
done
