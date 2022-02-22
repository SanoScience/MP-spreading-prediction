echo "### ALL ###"

python3 simulation_MAR.py ALL 8 30 0.5 2000000 2
echo "#"
python3 simulation_MAR.py ALL 8 30 0 2000000 2
echo "#"
python3 simulation_MAR.py ALL 8 30 1 2000000 2
echo "#"

echo "### AD ###"

python3 simulation_MAR.py AD 8 30 0.5 2000000 2
echo "#"
python3 simulation_MAR.py AD 8 30 0 2000000 2
echo "#"
python3 simulation_MAR.py AD 8 30 1 2000000 2
echo "#"

echo "### LMCI ###"

python3 simulation_MAR.py LMCI 8 30 0.5 2000000 2
echo "#"
python3 simulation_MAR.py LMCI 8 30 0 2000000 2
echo "#"
python3 simulation_MAR.py LMCI 8 30 1 2000000 2
echo "#"

echo "### EMCI ###"

python3 simulation_MAR.py EMCI 8 30 0.5 2000000 2
echo "#"
python3 simulation_MAR.py EMCI 8 30 0 2000000 2
echo "#"
python3 simulation_MAR.py EMCI 8 30 1 2000000 2
echo "#"

echo "### CN ###"

python3 simulation_MAR.py CN 8 30 0.5 2000000 2
echo "#"
python3 simulation_MAR.py CN 8 30 0 2000000 2
echo "#"
python3 simulation_MAR.py CN 8 30 1 2000000 2
echo "#"