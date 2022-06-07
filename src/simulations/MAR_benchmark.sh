# python3 simulation_MAR.py <category> <cores> <train_size> <lamda> <matrix> <use_binary> <iter_max> <N_fold>
echo "### ALL ###"
python3 simulation_MAR.py ALL 1 211 0 2 0 3000000 212
python3 simulation_MAR.py ALL 1 211 0 2 1 3000000 212
echo "### AD ###"
python3 simulation_MAR.py AD 1 23 0 2 0 3000000 24
python3 simulation_MAR.py AD 1 23 0 2 1 3000000 24
echo "### LMCI ###"
python3 simulation_MAR.py LMCI 1 46 0 2 0 3000000 47
python3 simulation_MAR.py LMCI 1 46 0 2 1 3000000 47
echo "### MCI ###"
python3 simulation_MAR.py MCI 1 29 0 2 0 3000000 30
python3 simulation_MAR.py MCI 1 29 0 2 1 3000000 30
echo "### EMCI ###"
python3 simulation_MAR.py EMCI 1 59 0 2 0 3000000 60
python3 simulation_MAR.py EMCI 1 59 0 2 1 3000000 60
echo "### CN ###"
python3 simulation_MAR.py CN 1 50 0 2 0 3000000 51
python3 simulation_MAR.py CN 1 50 0 2 1 3000000 51
