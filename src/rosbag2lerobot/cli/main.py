"""Click group and command registration for the rosbag2lerobot CLI."""

from __future__ import annotations

import click

from rosbag2lerobot.cli._common import _setup_logging
from rosbag2lerobot.cli.audit_timestamps import audit_timestamps
from rosbag2lerobot.cli.convert import convert
from rosbag2lerobot.cli.inspect import inspect
from rosbag2lerobot.cli.preview import preview_cmd
from rosbag2lerobot.cli.push_to_hub import push_to_hub_cmd
from rosbag2lerobot.cli.quality_report import quality_report_cmd
from rosbag2lerobot.cli.scaffold import scaffold
from rosbag2lerobot.cli.to_mcap import to_mcap
from rosbag2lerobot.cli.validate_config import validate_config
from rosbag2lerobot.cli.validate_dataset import validate_dataset_cmd
from rosbag2lerobot.cli.validate_msg import validate_msg


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """rosbag2lerobot – convert ROS2 rosbags to LeRobot Dataset v3.0."""
    _setup_logging(verbose)


main.add_command(convert)
main.add_command(inspect)
main.add_command(scaffold)
main.add_command(validate_config)
main.add_command(validate_dataset_cmd)
main.add_command(quality_report_cmd)
main.add_command(audit_timestamps)
main.add_command(validate_msg)
main.add_command(preview_cmd)
main.add_command(push_to_hub_cmd)
main.add_command(to_mcap)


if __name__ == "__main__":
    main()
