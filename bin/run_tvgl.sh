# dir setting
cd `dirname $0`
cd ..
cd exp


seeds_list=(
  '[[1,2,3],[4,5,6]]'
)

breaks_list=(
  '[[0, 100, 200, 300],[0, 150, 250, 300]]'
)


for seeds in "${seeds_list[@]}"; do
  for breaks in "${breaks_list[@]}"; do
    echo "Running with dims=($dims), seeds=$seeds, breaks=$breaks"
    python3 train_tvgl.py \
      --dims 5 5 \
      --breaks-json "$breaks" \
      --seeds-json "$seeds" \
      --T 300 \
      --alpha 0.01 \
      --beta 1.0 \
      --base-dir "../syn_l1" \
      --psi l1 \
      --samples-per-t 10
  done
done

