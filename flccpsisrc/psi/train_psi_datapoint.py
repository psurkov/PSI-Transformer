from omegaconf import DictConfig

from flccpsisrc.common.utils import run_with_config
from flccpsisrc.psi.psi_datapoint.psi_datapoint_facade import PSIDatapointFacade


def train(config: DictConfig) -> None:
    tree_loader = PSIDatapointFacade(config)
    tree_loader.train()


if __name__ == "__main__":
    run_with_config(train)
