# dir setting
cd `dirname $0`
cd ..
cd exp

#dims_list=("3 3" "5 5" "5 10" "10 10")
seeds_list=(
  #'[[1,2,3],[4,5,6]]'
  #'[[7,8,9],[10,11,12],[13,14,15]]'
  '[[13,14,15],[16,17,18]]'
  #'[[19,20,21],[22,23,24]]'
  #'[[25,26,27],[28,29,30]]'
)
breaks_list=(
  #'[[0, 100, 200, 300],[0, 100, 200, 300]]'
  '[[0, 100, 200, 300],[0, 150, 250, 300]]'
)
logs_list=(
  #'first'
  'latest'
)



for seeds in "${seeds_list[@]}"; do
    for breaks in "${breaks_list[@]}"; do
        for log in "${logs_list[@]}"; do
            python3 train_streamktvgl.py --dims 15 15 \
                --breaks-json "$breaks" \
                --seeds-json "$seeds" \
                --T 300 \
                --lambdas 0.05 0.05 \
                --rhos 1.5 1.5 \
                --init "Glasso" \
                --window-size 50 \
                --step-size 5 \
                --log "$log" \
                --base-dir "../syn_l1" \
                --psi l1
        done
    done
done
