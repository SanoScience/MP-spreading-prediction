# python3 simulation_GCN.py <category> <matrix> <epochs>
set TF_ENABLE_ONEDNN_OPTS=0
echo "ALL"
python3 simulation_GCN.py 	ALL 	0 	20000
python3 simulation_GCN.py 	ALL 	2	20000
python3 simulation_GCN.py 	ALL	    3	20000

echo "AD"
python3 simulation_GCN.py 	AD   	0       20000
python3 simulation_GCN.py 	AD   	2       20000
python3 simulation_GCN.py 	AD   	3       20000

echo "LMCI"
python3 simulation_GCN.py 	LMCI   	0       20000
python3 simulation_GCN.py 	LMCI   	2       20000
python3 simulation_GCN.py 	LMCI   	3       20000

echo "MCI"
python3 simulation_GCN.py 	MCI   	0       20000
python3 simulation_GCN.py 	MCI   	2       20000
python3 simulation_GCN.py 	MCI   	3       20000

echo "EMCI"
python3 simulation_GCN.py 	EMCI	0       20000
python3 simulation_GCN.py 	EMCI   	2       20000
python3 simulation_GCN.py 	EMCI   	3       20000

echo "CN"
python3 simulation_GCN.py 	CN   	0       20000
python3 simulation_GCN.py 	CN   	2       20000
python3 simulation_GCN.py 	CN   	3       20000
