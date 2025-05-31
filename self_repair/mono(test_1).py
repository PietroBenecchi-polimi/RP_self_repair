import pipeline as pipeline
import mc_opt_interface as mc
import json
import utils.generateData as gd
with open('data/hmtfactor_config.json', 'r') as file:
    factors = dict(json.load(file))

def run_oversampling_pipeline(n_data_to_verify, n_samples, data_type_second_validation: str, points_regressor, skip_cache = True):
    
    # Pipeline has 
    # - ground_truth_regressor: regressor trained on the full dataset
    # - regressor: regressor trained on a sample of the dataset
    # - test_set: dataset used for the validation(len = n_data_to_verify)
    p = pipeline.Pipeline(training_dataset_path = "data/dataset1000.csv",
                           test_data_path = "data/initial_configurations_to_improve.csv", points_regressor = points_regressor, n_data_to_verify = n_data_to_verify)


    # Comments
    opt_configs = mc.opt_optimization(p.test_set, p.regressor, f"regressor_{points_regressor}", skip_cache)
    ground_truth_first_test = mc.mc_results_from_configs(opt_configs.drop(columns=["SCS"]), p.ground_truth_regressor)
    invalid_configs, _ = p.validate_configurations(opt_configs, ground_truth_first_test)

    for _, config in invalid_configs.iterrows():
        generated_configs_from_invalid = gd.synthesize_data(config)
        ground_truth_generated_configs = mc.mc_results_from_configs(generated_configs_from_invalid, p.ground_truth_regressor)
        _, epsilon_array = p.validate_configurations(generated_configs_from_invalid, ground_truth_generated_configs)

    ## Oversmapling methods. On what are they based?