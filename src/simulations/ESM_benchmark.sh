# python3 simulation_ESM.py <category> <cores> <beta_0> <delta_0> <mu_noise> <sigma_noise> <iterations>
echo "### ALL ###"
python3 simulation_ESM.py ALL 10 0 0 0 0 5
python3 simulation_ESM.py ALL 10 0.5 0.5 0 0 5
python3 simulation_ESM.py ALL 10 1 1 0 0 5

python3 simulation_ESM.py ALL 10 0 0 1 1 5
python3 simulation_ESM.py ALL 10 0.5 0.5 1 1 5
python3 simulation_ESM.py ALL 10 1 1 1 1 5
echo "### AD ###"
python3 simulation_ESM.py AD 10 0 0 0 0 5
python3 simulation_ESM.py AD 10 0.5 0.5 0 0 5
python3 simulation_ESM.py AD 10 1 1 0 0 5

python3 simulation_ESM.py AD 10 0 0 1 1 5
python3 simulation_ESM.py AD 10 0.5 0.5 1 1 5
python3 simulation_ESM.py AD 10 1 1 1 1 5
echo "### LMCI ###"
python3 simulation_ESM.py LMCI 10 0 0 0 0 5
python3 simulation_ESM.py LMCI 10 0.5 0.5 0 0 5
python3 simulation_ESM.py LMCI 10 1 1 0 0 5

python3 simulation_ESM.py LMCI 10 0 0 1 1 5
python3 simulation_ESM.py LMCI 10 0.5 0.5 1 1 5
python3 simulation_ESM.py LMCI 10 1 1 1 1 5
echo "### MCI ###"
python3 simulation_ESM.py MCI 10 0 0 0 0 5
python3 simulation_ESM.py MCI 10 0.5 0.5 0 0 5
python3 simulation_ESM.py MCI 10 1 1 0 0 5

python3 simulation_ESM.py MCI 10 0 0 1 1 5
python3 simulation_ESM.py MCI 10 0.5 0.5 1 1 5
python3 simulation_ESM.py MCI 10 1 1 1 1 5
echo "### EMCI ###"
python3 simulation_ESM.py EMCI 10 0 0 0 0 5
python3 simulation_ESM.py EMCI 10 0.5 0.5 0 0 5
python3 simulation_ESM.py EMCI 10 1 1 0 0 5

python3 simulation_ESM.py EMCI 10 0 0 1 1 5
python3 simulation_ESM.py EMCI 10 0.5 0.5 1 1 5
python3 simulation_ESM.py EMCI 10 1 1 1 1 5
echo "### CN ###"
python3 simulation_ESM.py CN 10 0 0 0 0 5
python3 simulation_ESM.py CN 10 0.5 0.5 0 0 5
python3 simulation_ESM.py CN 10 1 1 0 0 5

python3 simulation_ESM.py CN 10 0 0 1 1 5
python3 simulation_ESM.py CN 10 0.5 0.5 1 1 5
python3 simulation_ESM.py CN 10 1 1 1 1 5
