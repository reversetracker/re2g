import logging.config

import pytest
import pytest_asyncio
import yaml

import directories

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_logging_configs(request):
    """Cleanup a testing directory once we are finished."""
    with open(directories.logging, "r") as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
        logging.config.dictConfig(config)


@pytest.fixture(scope="session", autouse=True)
def setup_lifecycle(request):
    def finalizer():
        logger.debug("Shutting down unittest session..")

    request.addfinalizer(finalizer)
