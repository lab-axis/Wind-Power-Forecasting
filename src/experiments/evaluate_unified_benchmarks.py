"""Republish a completed corrected run; never patch scores with historical constants."""
import argparse
from src.experiments.run_comprehensive_extended_matrix import publish_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Completed models/runs/corrected_v3_* directory")
    publish_run(parser.parse_args().run_dir)
