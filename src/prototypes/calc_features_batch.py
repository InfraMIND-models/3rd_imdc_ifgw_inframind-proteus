import glob
import os
from pathlib import Path

from inframind_proteus.outbreak_dynamics.utils import map_parallel_or_sequential


def main():
    # main_out_dir = Path("outputs/validation_round_calibration")
    main_out_dir = Path(".local/mind-runner_local/validation_round_calibration")

    # --- Regex
    # Try to extract year from output_dir name using the location-year format.
    # fmt = "{location_id}_{year}"
    # # Create a regex pattern from the format string.
    # pattern = fmt.replace("{location_id}", r"(?P<location_id>.+)")
    # pattern = pattern.replace("{year}", r"(?P<year>\d{4})")
    # # match = re.match(pattern, self.output_dir.name)
    #
    # pattern = "*_*"
    pattern = "??_[0-9][0-9][0-9][0-9]"  # More explicit
    subdirs = glob.glob(str(main_out_dir / pattern))


    def _task(subdir):
        cmd = "uv run calc-outbreak-features-from-posteriors"
        cmd += f" -o {subdir}"
        os.system(cmd)
        return subdir

    _contents = subdirs

    map_parallel_or_sequential(
        _task, _contents, ncpus=6
    )


if __name__ == "__main__":
    main()
