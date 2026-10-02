import argparse
from pathlib import Path
from tdpy.cli import add_plot_arguments

from tdpy.astro import run_target_visibility_diagnostic

'''
Estimate the visibility of a target on the sky from a given observatory, for a given night and across a given year.

The script uses TDpy's shared visibility calculation and figure writer.
'''

OBSERVATORIES = {
    'LCO': {
        'latiobvt': -29.01418,  # [deg]
        'longobvt': -70.69239,  # [deg]
        'heigobvt': 2515.819,  # [m]
        'offstimeobvt': 0.0,  # [hour]
    },
    'TUG': {
        'latiobvt': 36.824166,  # [deg]
        'longobvt': 30.335555,  # [deg]
        'heigobvt': 2500.0,  # [m]
        'offstimeobvt': 3.0,  # [hour]
    },
}

TARGETS = {
    'TOI-1233': {
        'right_ascension_degrees': 186.574,  # [deg]
        'declination_degrees': -51.363,  # [deg]
    },
    'TOI-700': {
        'right_ascension_degrees': 97.446,  # [deg]
        'declination_degrees': -65.579,  # [deg]
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Plot nightly and annual target visibility.',
    )
    parser.add_argument('--target', choices=TARGETS, default='TOI-1233')
    parser.add_argument('--observatory', choices=OBSERVATORIES, default='TUG')
    parser.add_argument('--night', default='2022-07-13 00:00:00')
    parser.add_argument('--year-start', default='2022-01-01 00:00:00')
    add_plot_arguments(parser)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    observatory = OBSERVATORIES[arguments.observatory]
    target = TARGETS[arguments.target]
    output_path = Path(__file__).resolve().parent / 'visuals' / (
        f'target_visibility_{arguments.target.lower()}.{arguments.typefileplot}'
    )
    run_target_visibility_diagnostic(
        output_path=output_path,
        target_label=arguments.target,
        observatory_label=arguments.observatory,
        right_ascension_degrees=target['right_ascension_degrees'],
        declination_degrees=target['declination_degrees'],
        latitude_degrees=observatory['latiobvt'],
        longitude_degrees=observatory['longobvt'],
        height_meters=observatory['heigobvt'],
        utc_offset_hours=observatory['offstimeobvt'],
        night=arguments.night,
        year_start=arguments.year_start,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

