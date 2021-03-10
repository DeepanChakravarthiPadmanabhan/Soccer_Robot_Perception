import logging
import click
import gin

from soccer_robot_perception.evaluate.evaluate_model import evaluate_model
from soccer_robot_perception.utils import setup_logging
from soccer_robot_perception.train import _register_configurables

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--config", "-c", help="config file for evaluation parameters")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]),
)
@click.option("--log-dir", default="")
@click.option("--model-path", required=True, default="model/model.pth")
@click.option("--report-path", required=True, default="report/")
def evaluate(config, log_level, log_dir, model_path, report_path):
    setup_logging(log_level=log_level, log_dir=log_dir)
    _register_configurables()
    gin.parse_config_file(config)
    LOGGER.info(">evaluating model")
    evaluate_model(model_path, report_path)


if __name__ == "__main__":
    evaluate()
