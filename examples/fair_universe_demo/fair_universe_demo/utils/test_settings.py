import argparse
import json
import os
from typing import Any, Dict

import numpy as np


def generate_test_settings_nominal(file_savepath: str = None) -> Dict[str, Any]:
    test_settings = {}
    test_settings["systematics"] = {
        "tes": False,
        "jes": False,
        "soft_met": False,
        "ttbar_scale": False,
        "diboson_scale": False,
        "bkg_scale": False,
    }
    test_settings["num_pseudo_experiments"] = 20
    test_settings["num_of_sets"] = 10
    test_settings["ground_truth_mus"] = np.linspace(0.1, 3, test_settings["num_of_sets"]).tolist()
    test_settings["random_mu"] = False

    if file_savepath:
        with open(file_savepath, "w") as f:
            json.dump(test_settings, f, indent=4)

    print(f"Wrote `nominal` version to {file_savepath}")
    return test_settings


def generate_test_settings_systematics(file_savepath: str = None) -> Dict[str, Any]:
    test_settings = {}
    test_settings["systematics"] = {
        "tes": True,
        "jes": True,
        "soft_met": True,
        "ttbar_scale": True,
        "diboson_scale": True,
        "bkg_scale": True,
    }
    test_settings["num_pseudo_experiments"] = 20
    test_settings["num_of_sets"] = 10
    test_settings["ground_truth_mus"] = np.linspace(0.1, 3, test_settings["num_of_sets"]).tolist()
    test_settings["random_mu"] = False

    if file_savepath:
        with open(file_savepath, "w") as f:
            json.dump(test_settings, f, indent=4)

    print(f"Wrote `systematics` version to {file_savepath}")
    return test_settings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test settings for model evaluation after training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nominal",
        type=bool,
        default=False,
        help="Whether to generate test settings for the nominal case or with systematics (default)",
    )
    args = parser.parse_args()

    if args.nominal:
        generate_test_settings_nominal(f"{os.path.dirname(__file__)}/../../conf/test_settings_nominal.json")
    else:
        generate_test_settings_systematics(f"{os.path.dirname(__file__)}/../../conf/test_settings.json")
