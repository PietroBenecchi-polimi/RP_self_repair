Human-Robot Interactive Scenario Uppaal Model Generator 
====================================

The repository contains the implementation of an Uppaal model generator for the analysis of human-robot interaction scenarios in service settings.
The model consists of a Stochastic Hybrid Automata network, described in the following articles: 

- Lestingi, L., Sbrolli, C., Scarmozzino, P., Romeo, G., Bersani, M. M., & Rossi, M. (2022). Formal modeling and verification of multi-robot interactive scenarios in service settings. In FormaliSE 2022 (pp. 80-90). [doi.org/10.1145/3524482.3527653][paper5]
- Lestingi, L., Askarpour, M., Bersani, M. M., & Rossi, M. (2021). A Deployment Framework for Formally Verified Human-Robot Interactions. IEEE Access. [doi.org/10.1109/ACCESS.2021.3117852][paper4]
- Lestingi, L., Askarpour, M., Bersani, M. M., & Rossi, M. (2020, September). *Formal verification of human-robot interaction in healthcare scenarios*. In SEFM 2020 (pp. 303-324). Springer, Cham. [doi.org/10.1007/978-3-030-58768-0_17][paper2]
- Lestingi, L., Askarpour, M., Bersani, M. M., & Rossi, M. (2020). *A Model-driven Approach for the Formal Analysis of Human-Robot Interaction Scenarios*. In SMC 2020 (pp. 1907-1914), IEEE. [doi.org/10.1109/SMC42975.2020.9283204][paper3]
- Lestingi, L., Askarpour, M., Bersani, M. M., & Rossi, M. (2020). *Statistical Model Checking of Human-Robot Interaction Scenarios*. AREA 2020 [doi.org/10.4204/EPTCS.319.2][paper1].

The network features automata modeling the *humans*, the *robot*, and the *robot controller*.

It is possible to customize the scenario parameters (e.g., how many humans, their features, the layout shape, etc.) in a JSON file. Example JSON files can be found in the [`./resources/input_params/`](resources/input_params) folder.
The tool customizes the templates constituting the Uppaal model (one for each automaton in the network, contained in folder [`./resources/upp_templates/`](resources/upp_templates)) based on such user-specified parameters. The generated models are saved in the [`./resources/gen_models/`](resources/gen_models) folder.

The tool allows users to estimate the probability of success of the scenario through SMC experiments. Once the model has been generated, the tool runs the [verify.sh](resources/upp_resources) script to launch the verification experiment. Uppaal verification output and results can be found in the [`./resources/upp_results/`](resources/upp_results) folder.

Authors:

| Name              | E-mail address           |
|:----------------- |:-------------------------|
| Lestingi Livia    | livia.lestingi@polimi.it |


Configuration File Setup
-----------

The [main script](src/main.py) requires as input parameter the path to a configuration file, whose template can be found within the [`./resources/config/`](resources/config) folder.

Make sure that the following properties match your environment: 
- **UPPAAL_PATH** is the path to Uppaal [command line utility][verifyta];

**Note**: The tool has been tested on Uppaal **v.4.1.24** on Mac OS X. Should you run into any issue while testing with a different configuration please report to livia.lestingi@polimi.it.

Python Dependencies
-----------

Install the required dependencies:

	pip install -r $REPO_PATH/requirements.txt

Add the repo path to your Python path (fixes ModuleNotFoundError while trying to execute from command line):

	export PYTHONPATH="${PYTHONPATH}:$REPO_PATH"


Main Script's Input Parameters
-----------

Run the main script specifying the name of the scenario to be analysed (there must be a $SCENARIO_NAME.json file within the [`./resources/input_params/`](resources/input_params) folder):

	python3 $REPO_PATH/src/main.py $SCENARIO_NAME
	
---

*Copyright &copy; 2021 Livia Lestingi*

[paper1]: https://doi.org/10.4204/EPTCS.319.2
[paper2]: https://doi.org/10.1007/978-3-030-58768-0_17
[paper3]: https://doi.org/10.1109/SMC42975.2020.9283204
[paper4]: https://doi.org/10.1109/ACCESS.2021.3117852
[paper5]: https://doi.org/10.1145/3524482.3527653
[paper6]: https://doi.org/10.1109/MIS.2022.3215698
[angluin]: https://doi.org/10.1016/0890-5401(87)90052-6
[uppaal]: https://uppaal.org/
[dep]: https://github.com/LesLivia/hri_deployment
[verifyta]: https://docs.uppaal.org/toolsandapi/verifyta/