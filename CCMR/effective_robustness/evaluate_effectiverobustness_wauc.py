'''
Small script to compute the effective robustness
of a model by directly providing the WAUC on the ID and OOD datasets.
'''
import json
import os
import argparse

#from helper_functions import baselines
import baselines
#from helper_functions.ownutilities import args_to_outputfilepath


def create_mini_parser():
    parser = argparse.ArgumentParser(usage='%(prog)s [options (see below)]')
    
    parser.add_argument('--id-wauc', type=float,
                        help="WAUC on the ID dataset")
    parser.add_argument('--ood-wauc', type=float,
                        help="WAUC on the OOD dataset")
        
    parser.add_argument('--baseline_file', type=str,
                        help="file path where to store baseline parameters, defaults to '{output_folder}/baseline_{dataset_id}_{dataset_ood}.json'")
        
    return parser

if __name__ == '__main__':
    parser = create_mini_parser()
    args = parser.parse_args()

    id_wauc = args.id_wauc

    ood_wauc = args.ood_wauc

    #
    # load baseline
    #
    baseline_file = args.baseline_file
    bl = baselines.BaselineLogitLinearRegression.deserialize_json(baseline_file)

    #
    # evaluate effective robustness
    #
    beta_wauc = bl.predict(id_wauc)[0]
    er = ood_wauc - beta_wauc

    print(f'Effective Robustness: ER_WAUC = {er:.6f}')

    if er > 0:
        print("Congratz, you're above average!")
    else:
        print("This indicates bad generalization.")
