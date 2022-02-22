echo "### ALL ###"

# python3 simulation_ESM.py ALL 8 0 0 1 1
python3 simulation_ESM.py ALL 8 0.5 0.5 1 1
python3 simulation_ESM.py ALL 8 1 1 1 1

echo "### AD ###"

python3 simulation_ESM.py AD 8 0 0 1 1
python3 simulation_ESM.py AD 8 0.5 0.5 1 1
python3 simulation_ESM.py AD 8 1 1 1 1

echo "### LMCI ###"

python3 simulation_ESM.py LMCI 8 0 0 1 1
python3 simulation_ESM.py LMCI 8 0.5 0.5 1 1
python3 simulation_ESM.py LMCI 8 1 1 1 1

echo "### EMCI ###"

python3 simulation_ESM.py EMCI 8 0 0 1 1
python3 simulation_ESM.py EMCI 8 0.5 0.5 1 1
python3 simulation_ESM.py EMCI 8 1 1 1 1

echo "### CN ###"

python3 simulation_ESM.py CN 8 0 0 1 1
python3 simulation_ESM.py CN 8 0.5 0.5 1 1
python3 simulation_ESM.py CN 8 1 1 1 1