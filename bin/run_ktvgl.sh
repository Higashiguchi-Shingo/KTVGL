# dir setting
cd `dirname $0`
cd ..
cd exp

dims_list=("3 3" "5 5" "5 10" "10 10")
seeds_list=(
  #'[[1,2,3],[4,5,6]]'
  #'[[7,8,9],[10,11,12]]'
  #'[[13,14,15],[16,17,18],[19,20,21]]'
  '[[19,20,21],[22,23,24],[25,26,27]]'
  #'[[25,26,27],[28,29,30],[31,32,33]]'
)
#seeds_list=(
#  '[[1,2,3],[4,5,6]]'
#  '[[7,8,9],[10,11,12]]'
#  '[[13,14,15],[16,17,18]]'
#  '[[19,20,21],[22,23,24]]'
#  '[[25,26,27],[28,29,30]]'
#)
breaks_list=(
  #'[[0, 100, 200, 300],[0, 100, 200, 300], [0, 100, 200, 300]]'
  '[[0, 100, 200, 300],[0, 150, 250, 300], [0, 100, 250, 300]]'
  #'[[0, 100, 200, 300],[0, 150, 250, 300]]'
)


#for seeds in "${seeds_list[@]}"; do
# for breaks in "${breaks_list[@]}"; do
#    echo "Running with dims=($dims), seeds=$seeds, breaks=$breaks"
#    python3 train_ktvgl.py \
#      --dims 3 3 \
#      --breaks-json "$breaks" \
#      --seeds-json "$seeds" \
#     --T 300 \
#      --base-dir "../exp_results" \
#     --lambda 0.05 0.05 \
#      --rho 2.0 2.0
# done
#done

#for seeds in "${seeds_list[@]}"; do
#  for breaks in "${breaks_list[@]}"; do
#    echo "Running with dims=($dims), seeds=$seeds, breaks=$breaks"
#    python3 train_ktvgl.py \
#      --dims 5 5 \
#      --breaks-json "$breaks" \
#      --seeds-json "$seeds" \
#      --T 300 \
#      --base-dir "../exp_results" \
#      --lambda 0.05 0.05 \
#      --rho 1.5 1.5
#  done
#done

for seeds in "${seeds_list[@]}"; do
  for breaks in "${breaks_list[@]}"; do
    echo "Running with dims=($dims), seeds=$seeds, breaks=$breaks"
    python3 train_ktvgl.py \
      --dims 15 15 15 \
      --breaks-json "$breaks" \
      --seeds-json "$seeds" \
      --T 300 \
      --base-dir "../syn_l1" \
      --lambda 0.03 0.03 0.03 \
      --rho 2.0 2.0 2.0 \
      --gauge "trace" \
      --init "Glasso" \
      --psi l1
  done
done


#for seeds in "${seeds_list[@]}"; do
 # for breaks in "${breaks_list[@]}"; do
  #  echo "Running with dims=($dims), seeds=$seeds, breaks=$breaks"
   # python3 train_ktvgl.py \
    #  --dims 10 10 \
     # --breaks-json "$breaks" \
     # --seeds-json "$seeds" \
     #--T 300 \
    # --base-dir "../exp_results" \
    # --lambda 0.03 0.03 \
    #--rho 2.0 2.0
#  done
#done
