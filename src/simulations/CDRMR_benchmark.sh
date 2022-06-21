# python3 simulation_CDRMR.py <category> <cores> <train_size> <iter_max> <N_fold>

echo "ALL"
python3 simulation_CDRMR.py 	ALL	16 	211 	3000000 	212

echo "AD"
python3 simulation_CDRMR.py 	AD 	16 	23 	3000000 	24

echo "LMCI"
python3 simulation_CDRMR.py 	LMCI 	16 	46 	3000000 	47

echo "MCI"
python3 simulation_CDRMR.py 	MCI 	16 	29 	3000000 	30

echo "EMCI"
python3 simulation_CDRMR.py 	EMCI 	16 	59 	3000000 	60

echo "CN"
python3 simulation_CDRMR.py 	CN 	16 	50 	3000000 	51
