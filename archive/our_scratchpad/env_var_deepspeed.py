import os
import general_utils

filtered = {k: v for k, v in os.environ.items() if "rank" in k.lower()}
general_utils.print_dict(filtered)
