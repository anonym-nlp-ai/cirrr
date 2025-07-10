from argparse import ArgumentParser
from datetime import datetime
import sys

sys.path.append("./../")
from aimw.app.topology.graph.cir3_graph import Cir3Graph
from aimw.app.utils import common_utils
from loguru import logger
from aimw.app.core.log_config import LoggingSettings, get_log_settings, setup_app_logging


# setup logging as early as possible
setup_app_logging(config=LoggingSettings())

def main(args):

    output_dir = args.output_dir
    cir3Graph = Cir3Graph()

    document = "A mortgage is a loan used to purchase property, with the property serving as collateral.\
          It's a liability that typically requires monthly payments of principal and interest.\
        An annuity is a financial product that provides a guaranteed income stream, often used for retirement. \
        It can be purchased with a lump sum or through regular payments. \
        Taxes are mandatory financial charges imposed by the government on income, property, or goods. \
        They fund public services and can impact investment decisions. \
        Life insurance is a contract that provides financial protection to beneficiaries upon the insured person's death. \
        Premiums are paid, and the payout can help cover expenses or replace lost income. \
        A pension is a retirement plan that provides regular income after retirement. \
        It can be employer-sponsored or privately funded. The relationship between these concepts lies in financial planning. \
        Mortgages and annuities represent long-term financial commitments, while taxes affect their costs and returns. \
        Life insurance can protect mortgage payments or provide funds for annuity purchases. \
        Pensions and annuities are income sources for retirement, with taxes impacting their value. \
        Overall, these financial instruments interact to shape an individual's financial stability \
        throughout their life, particularly during retirement."

    logger.info("Starting CIR3 ...")
    # Params can be adjusted as part of the input:
    inputs = {"document": document, "num_steps": 0, "M": 3, "N": 5, "L": 3, "K": 4}
    output = cir3Graph.topology.invoke(inputs)
    logger.info(f"Final set of question-answer pairs: \n {output['final_qas']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cir3_{timestamp}.json"
    common_utils.save_json(output["final_qas"], output_dir, filename)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/cir3",
        help="Directory to store the outputs.",
    )
    main(parser.parse_args())
