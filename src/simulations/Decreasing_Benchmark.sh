# python3 simulation_ESM.py <category> <cores> <beta_0> <delta_0> <mu_noise> <sigma_noise>
# python3 simulation_NDM.py <category> <cores> <beta>
#python3 simulation_MAR.py <category> <cores> <train_size> <lamda> <matrix> <use_binary> <iter_max> <N_fold>
# python3 simulation_GCN.py <category> <matrix> <epochs>

echo "ESM"
python3 simulation_ESM.py   Decreasing    16  0   0   0   0

echo "NDM"
python3 simulation_NDM.py   Decreasing    16  0.2

echo "MAR"
python3 simulation_MAR.py   Decreasing    16  99  0   3   1 3000000 100

echo "GCN"
python3 simulation_GCN.py   Decreasing    0  2000
