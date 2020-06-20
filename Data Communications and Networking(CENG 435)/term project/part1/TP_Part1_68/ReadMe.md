2237899-Ahmet Dara VEFA && 2169589-Alper KOCAMAN
term project folder has 2 sub folders: discoveryScripts and experimentScripts. Before starting anything you should syncronize all of your nodes by running 'sudo ntpdate -u pool.ntp.org' command on each node.
	discoveryScripts has 2 shell scripts named configureR1.sh and configureR2.sh; and 5 python scripts named r1.py, r2.py, r3.py, destination.py and source.py and a subfolder named Link Costs. The shell scripts need to be ran in their respective nodes with ./'shellScriptName' before finding RTTs. Note that you may need to do chmod +x 'shellScriptName' before executing the shell script since you may not have executive rights. After running the shell scripts you can run the python codes in their respective nodes with python3 'name of the python script'. You should first run source.py then destination.py then r2.py then the rest of the python scripts for correctly finding the end to end delays. The clients will print the delays(for 1000 messages) in the console, you can either copy them or insert them to a file by running the python code like python3 r3.py > delays.txt. Or you can look at 'targetNodeName' + "-" + 'currentNodeName' + "_link_cost.txt" for the average delay.
		Link Costs folder has link costs of each connection as a txt file.
	experimentScripts has 3 python scripts named destinationExperiment.py r3Experiment.py sourceExperiment.py and 2 subfolders named experimentdelayscripts and experimentResults. You need to run destinationExperiment.py then r3Experiment.py then sourceExperiment.py in that order with python3 'pythonScriptName' after adding the delay for that experiment.
		experimentDelayScripts has the shell scripts for source and r3 nodes for each of the 3 experiments. You should delete the delay(if there was a delay added before) and add a new one before starting each of the experiments in each of the nodes. You need to run ifconfig to find the interface of the link you want to delete: the ipv4 of the interface is the same as the ipv4 from the current node to the target node. Then you would delete the delay with sudo tc qdisc del dev 'interface' root. An example is sudo tc qdisc show dev eth1 root. Then you need to add a new delay with running the respective shell script of the experiment you are running with as ./'shellScriptName'. Again, you may need to give executive rights to yourself before running the shell scripts.
		Experiment results has 4 txt documents each containing the message delays(of 1000 messages) of their respective experiments. First experiment was done twice so there is 2 for that.