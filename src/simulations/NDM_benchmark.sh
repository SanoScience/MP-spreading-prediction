# python3 simulation_NDM.py <category> <cores> <beta>

echo "### ALL ###"
python3 simulation_NDM.py ALL 10 0.0 
python3 simulation_NDM.py ALL 10 0.1 
python3 simulation_NDM.py ALL 10 0.2 
python3 simulation_NDM.py ALL 10 0.4 
python3 simulation_NDM.py ALL 10 0.8 
python3 simulation_NDM.py ALL 10 1 

exit # The remaining simulations are already comprised in the "ALL" category

echo "### AD ###"
python3 simulation_NDM.py AD 10 0.0 
python3 simulation_NDM.py AD 10 0.1 
python3 simulation_NDM.py AD 10 0.2 
python3 simulation_NDM.py AD 10 0.4 
python3 simulation_NDM.py AD 10 0.8 
python3 simulation_NDM.py AD 10 1 
echo "### LMCI ###"
python3 simulation_NDM.py LMCI 10 0.0 
python3 simulation_NDM.py LMCI 10 0.1 
python3 simulation_NDM.py LMCI 10 0.2 
python3 simulation_NDM.py LMCI 10 0.4 
python3 simulation_NDM.py LMCI 10 0.8 
python3 simulation_NDM.py LMCI 10 1 
echo "### MCI ###"
python3 simulation_NDM.py MCI 10 0.0 
python3 simulation_NDM.py MCI 10 0.1 
python3 simulation_NDM.py MCI 10 0.2 
python3 simulation_NDM.py MCI 10 0.4 
python3 simulation_NDM.py MCI 10 0.8 
python3 simulation_NDM.py MCI 10 1 
echo "### EMCI ###"
python3 simulation_NDM.py EMCI 10 0.0 
python3 simulation_NDM.py EMCI 10 0.1 
python3 simulation_NDM.py EMCI 10 0.2 
python3 simulation_NDM.py EMCI 10 0.4 
python3 simulation_NDM.py EMCI 10 0.8 
python3 simulation_NDM.py EMCI 10 1 
echo "### CN ###"
python3 simulation_NDM.py CN 10 0.0 
python3 simulation_NDM.py CN 10 0.1 
python3 simulation_NDM.py CN 10 0.2 
python3 simulation_NDM.py CN 10 0.4 
python3 simulation_NDM.py CN 10 0.8 
python3 simulation_NDM.py CN 10 1 