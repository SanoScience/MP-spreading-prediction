# python3 simulation_MAR.py <category> <cores> <train_size> <lamda> <matrix> <use_binary> <iter_max> <N_fold>
echo "### ALL ###"
python3 simulation_MAR.py ALL 4 211 0 0 0 3000000 212
python3 simulation_MAR.py ALL 4 211 0 0 1 3000000 212

python3 simulation_MAR.py ALL 4 211 0 2 0 3000000 212
python3 simulation_MAR.py ALL 4 211 0 2 1 3000000 212

python3 simulation_MAR.py ALL 4 211 0 3 0 3000000 212
python3 simulation_MAR.py ALL 4 211 0 3 1 3000000 212

echo "### AD ###"
python3 simulation_MAR.py AD 4 23 0 0 0 3000000 24
python3 simulation_MAR.py AD 4 23 0 0 1 3000000 24

python3 simulation_MAR.py AD 4 23 0 2 0 3000000 24
python3 simulation_MAR.py AD 4 23 0 2 1 3000000 24

python3 simulation_MAR.py AD 4 23 0 3 0 3000000 24
python3 simulation_MAR.py AD 4 23 0 3 1 3000000 24
echo "### LMCI ###"
python3 simulation_MAR.py LMCI 4 46 0 0 0 3000000 47
python3 simulation_MAR.py LMCI 4 46 0 0 1 3000000 47

python3 simulation_MAR.py LMCI 4 46 0 2 0 3000000 47
python3 simulation_MAR.py LMCI 4 46 0 2 1 3000000 47

python3 simulation_MAR.py LMCI 4 46 0 3 0 3000000 47
python3 simulation_MAR.py LMCI 4 46 0 3 1 3000000 47
echo "### MCI ###"
python3 simulation_MAR.py MCI 4 29 0 0 0 3000000 30
python3 simulation_MAR.py MCI 4 29 0 0 1 3000000 30

python3 simulation_MAR.py MCI 4 29 0 2 0 3000000 30
python3 simulation_MAR.py MCI 4 29 0 2 1 3000000 30

python3 simulation_MAR.py MCI 4 29 0 3 0 3000000 30
python3 simulation_MAR.py MCI 4 29 0 3 1 3000000 30
echo "### EMCI ###"
python3 simulation_MAR.py EMCI 4 59 0 0 0 3000000 60
python3 simulation_MAR.py EMCI 4 59 0 0 1 3000000 60

python3 simulation_MAR.py EMCI 4 59 0 2 0 3000000 60
python3 simulation_MAR.py EMCI 4 59 0 2 1 3000000 60

python3 simulation_MAR.py EMCI 4 59 0 3 0 3000000 60
python3 simulation_MAR.py EMCI 4 59 0 3 1 3000000 60
echo "### CN ###"
python3 simulation_MAR.py CN 4 50 0 0 0 3000000 51
python3 simulation_MAR.py CN 4 50 0 0 1 3000000 51

python3 simulation_MAR.py CN 4 50 0 2 0 3000000 51
python3 simulation_MAR.py CN 4 50 0 2 1 3000000 51

python3 simulation_MAR.py CN 4 50 0 3 0 3000000 51
python3 simulation_MAR.py CN 4 50 0 3 1 3000000 51
