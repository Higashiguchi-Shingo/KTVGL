# dir setting
cd `dirname $0`
cd ..
cd exp

alpha_list=(0.01 0.02 0.03 0.04 0.05)
beta_list=(1.0 1.5 2.0 2.5 3.0)
dims_list=("3 3" "5 5" "5 10" "10 10")

for dims in "${dims_list[@]}"; do
  for alpha in "${alpha_list[@]}"; do
    for beta in "${beta_list[@]}"; do
      echo "Running with dims=($dims), alpha=$alpha, beta=$beta"
      python3 train_tvgl.py \
        --dims $dims \
        --alpha $alpha \
        --beta $beta \
        --breaks-json '[[0, 100, 200, 300],[0, 100, 200, 300]]' \
        --seeds-json '[[41,42,43],[44,45,46]]' \
        --T 300 \
        --base-dir "../grid"
    done
  done
done