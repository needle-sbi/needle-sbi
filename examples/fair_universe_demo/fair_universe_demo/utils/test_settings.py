import json
import os
from typing import Any, Dict

import numpy as np


def generate_test_settings(file_savepath: str = None) -> Dict[str, Any]:
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
    random_state = np.random.RandomState(42)
    test_settings["ground_truth_mus"] = (random_state.uniform(0.1, 3, test_settings["num_of_sets"])).tolist()
    test_settings["random_mu"] = True

    if file_savepath:
        with open(file_savepath, "w") as f:
            json.dump(test_settings, f, indent=4)

    return test_settings


if __name__ == "__main__":
    generate_test_settings(f"{os.path.dirname(__file__)}/../../conf/test_settings.json")
