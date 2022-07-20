# python3 simulation_ESM.py <category> <cores> <beta_0> <delta_0> <mu_noise> <sigma_noise>
echo "### ALL ###"
python3 simulation_ESM.py ALL 10 0 0 0 0
python3 simulation_ESM.py ALL 10 0.5 0.5 0 0
python3 simulation_ESM.py ALL 10 1 1 0 0

python3 simulation_ESM.py ALL 10 0 0 1 1
python3 simulation_ESM.py ALL 10 0.5 0.5 1 1
python3 simulation_ESM.py ALL 10 1 1 1 1

exit # The remaining simulations are already comprised in the "ALL" category

echo "### AD ###"
python3 simulation_ESM.py AD 10 0 0 0 0
python3 simulation_ESM.py AD 10 0.5 0.5 0 0
python3 simulation_ESM.py AD 10 1 1 0 0

python3 simulation_ESM.py AD 10 0 0 1 1
python3 simulation_ESM.py AD 10 0.5 0.5 1 1
python3 simulation_ESM.py AD 10 1 1 1 1
echo "### LMCI ###"
python3 simulation_ESM.py LMCI 10 0 0 0 0
python3 simulation_ESM.py LMCI 10 0.5 0.5 0 0
python3 simulation_ESM.py LMCI 10 1 1 0 0

python3 simulation_ESM.py LMCI 10 0 0 1 1
python3 simulation_ESM.py LMCI 10 0.5 0.5 1 1
python3 simulation_ESM.py LMCI 10 1 1 1 1
echo "### MCI ###"
python3 simulation_ESM.py MCI 10 0 0 0 0
python3 simulation_ESM.py MCI 10 0.5 0.5 0 0
python3 simulation_ESM.py MCI 10 1 1 0 0

python3 simulation_ESM.py MCI 10 0 0 1 1
python3 simulation_ESM.py MCI 10 0.5 0.5 1 1
python3 simulation_ESM.py MCI 10 1 1 1 1
echo "### EMCI ###"
python3 simulation_ESM.py EMCI 10 0 0 0 0
python3 simulation_ESM.py EMCI 10 0.5 0.5 0 0
python3 simulation_ESM.py EMCI 10 1 1 0 0

python3 simulation_ESM.py EMCI 10 0 0 1 1
python3 simulation_ESM.py EMCI 10 0.5 0.5 1 1
python3 simulation_ESM.py EMCI 10 1 1 1 1
echo "### CN ###"
python3 simulation_ESM.py CN 10 0 0 0 0
python3 simulation_ESM.py CN 10 0.5 0.5 0 0
python3 simulation_ESM.py CN 10 1 1 0 0

python3 simulation_ESM.py CN 10 0 0 1 1
python3 simulation_ESM.py CN 10 0.5 0.5 1 1
python3 simulation_ESM.py CN 10 1 1 1 1