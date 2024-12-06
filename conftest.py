"""Pytest fixtures for doctests."""

import pytest

import dkpy


@pytest.fixture(autouse=True)
def add_dkpy(doctest_namespace):
    """Add ``dkpy`` to namespace."""
    doctest_namespace["dkpy"] = dkpy


@pytest.fixture(autouse=True)
def add_example_scherer1997_p907(doctest_namespace):
    """Add generalized plant from [SGC97]_, Example 7 (p. 907)."""
    eg = dkpy.example_scherer1997_p907()
    doctest_namespace["example_scherer1997_p907"] = (
        eg["P"],
        eg["n_y"],
        eg["n_u"],
    )


@pytest.fixture(autouse=True)
def add_example_skogestad2006_p325(doctest_namespace):
    """Add generalized plant from [SP06]_, Table 8.1 (p. 325)."""
    eg = dkpy.example_skogestad2006_p325()
    doctest_namespace["example_skogestad2006_p325"] = (
        eg["P"],
        eg["n_y"],
        eg["n_u"],
        eg["K"],
    )
