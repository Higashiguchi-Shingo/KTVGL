# dir setting
cd `dirname $0`
cd ..
cd exp

#lambda_list=("0.01 0.01" "0.03 0.03" "0.05 0.05")
#rho_list=("1.0 1.0" "1.5 1.5" "2.0 2.0" "2.5 2.5" "3.0 3.0")
dims_list=("15 15")
lam1_list=("0.01" "0.03" "0.05")
lam2_list=("0.01" "0.03" "0.05")
rho1_list=("1.0" "1.5" "2.0")
rho2_list=("1.0" "1.5" "2.0")
init_methods=("Glasso")

for dims in "${dims_list[@]}"; do
  for lambda1 in "${lam1_list[@]}"; do
    for lambda2 in "${lam2_list[@]}"; do
      lambda="$lambda1 $lambda2"
      for rho1 in "${rho1_list[@]}"; do
        for rho2 in "${rho2_list[@]}"; do
          rho="$rho1 $rho2"
          for init_method in "${init_methods[@]}"; do
            echo "Running with dims=($dims), lambda=$lambda, rho=$rho, init=$init_method"
            python3 train_ktvgl.py \
              --dims $dims \
              --lambda $lambda \
              --rho $rho \
              --breaks-json '[[0, 100, 200, 300],[0, 100, 200, 300]]' \
              --seeds-json '[[41,42,43],[44,45,46]]' \
              --T 300 \
              --init $init_method \
              --base-dir "../grid_full"
          done
        done
      done
    done
  done
done