# python3 simulation_ESM.py <category> <cores> <beta_0> <delta_0> <mu_noise> <sigma_noise> <iterations>
echo "### ALL ###"
python3 simulation_ESM.py ALL 1 0 0 0 0 5
python3 simulation_ESM.py ALL 1 0.5 0.5 0 0 5
python3 simulation_ESM.py ALL 1 1 1 0 0 5

python3 simulation_ESM.py ALL 1 0 0 1 1 5
python3 simulation_ESM.py ALL 1 0.5 0.5 1 1 5
python3 simulation_ESM.py ALL 1 1 1 1 1 5
echo "### AD ###"
python3 simulation_ESM.py AD 1 0 0 0 0 5
python3 simulation_ESM.py AD 1 0.5 0.5 0 0 5
python3 simulation_ESM.py AD 1 1 1 0 0 5

python3 simulation_ESM.py AD 1 0 0 1 1 5
python3 simulation_ESM.py AD 1 0.5 0.5 1 1 5
python3 simulation_ESM.py AD 1 1 1 1 1 5
echo "### LMCI ###"
python3 simulation_ESM.py LMCI 1 0 0 0 0 5
python3 simulation_ESM.py LMCI 1 0.5 0.5 0 0 5
python3 simulation_ESM.py LMCI 1 1 1 0 0 5

python3 simulation_ESM.py LMCI 1 0 0 1 1 5
python3 simulation_ESM.py LMCI 1 0.5 0.5 1 1 5
python3 simulation_ESM.py LMCI 1 1 1 1 1 5
echo "### EMCI ###"
python3 simulation_ESM.py EMCI 1 0 0 0 0 5
python3 simulation_ESM.py EMCI 1 0.5 0.5 0 0 5
python3 simulation_ESM.py EMCI 1 1 1 0 0 5

python3 simulation_ESM.py EMCI 1 0 0 1 1 5
python3 simulation_ESM.py EMCI 1 0.5 0.5 1 1 5
python3 simulation_ESM.py EMCI 1 1 1 1 1 5
echo "### CN ###"
python3 simulation_ESM.py CN 1 0 0 0 0 5
python3 simulation_ESM.py CN 1 0.5 0.5 0 0 5
python3 simulation_ESM.py CN 1 1 1 0 0 5

python3 simulation_ESM.py CN 1 0 0 1 1 5
python3 simulation_ESM.py CN 1 0.5 0.5 1 1 5
python3 simulation_ESM.py CN 1 1 1 1 1 5
