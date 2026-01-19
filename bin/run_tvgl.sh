# dir setting
cd `dirname $0`
cd ..
cd exp

dims_list=("3 3" "5 5" "5 10" "10 10")
#seeds_list=(
#  '[[1,2,3],[4,5,6]]'
#  '[[7,8,9],[10,11,12]]'
#  '[[13,14,15],[16,17,18]]'
#  '[[19,20,21],[22,23,24]]'
#  '[[25,26,27],[28,29,30]]'
#)
seeds_list=(
  '[[1,2,3],[4,5,6],[7,8,9]]'
  '[[7,8,9],[10,11,12],[13,14,15]]'
  '[[13,14,15],[16,17,18],[19,20,21]]'
  '[[19,20,21],[22,23,24],[25,26,27]]'
  '[[25,26,27],[28,29,30],[31,32,33]]'
)
breaks_list=(
  '[[0, 100, 200, 300],[0, 100, 200, 300],[0, 100, 200, 300]]'
  #'[[0, 100, 200, 300],[0, 150, 250, 300]]'
)


for seeds in "${seeds_list[@]}"; do
  for breaks in "${breaks_list[@]}"; do
    echo "Running with dims=($dims), seeds=$seeds, breaks=$breaks"
    python3 train_tvgl.py \
      --dims 5 5 5 \
      --breaks-json "$breaks" \
      --seeds-json "$seeds" \
      --T 300 \
      --alpha 0.01 \
      --beta 2.0 \
      --base-dir "../syn_not_scale"
  done
done

