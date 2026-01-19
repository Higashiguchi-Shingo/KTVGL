# dir setting
cd `dirname $0`
cd ..
cd exp


dims_list=("3 3" "5 5" "5 10" "10 10")
lam1_list=("0.01" "0.03" "0.05")
lam2_list=("0.01" "0.03" "0.05")



for dims in "${dims_list[@]}"; do
  for lambda1 in "${lam1_list[@]}"; do
    for lambda2 in "${lam2_list[@]}"; do
        lambda="$lambda1 $lambda2"
        echo "Running with dims=($dims), lambda=$lambda"
        python3 train_kgl.py \
            --dims $dims \
            --lambda $lambda \
            --breaks-json '[[0, 100, 200, 300],[0, 100, 200, 300]]' \
            --seeds-json '[[41,42,43],[44,45,46]]' \
            --T 300 \
            --base-dir "../grid_full"
    done
  done
done