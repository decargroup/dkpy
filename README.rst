.. role:: class(code)

dkpy
====

``dkpy`` is a `D-K iteration <https://doi.org/10.1109/ACC.1994.735077>`_
library written in Python, aiming to build upon
`python-control <https://github.com/python-control/python-control>`_.

The package is currently a work-in-progress, and no API stability guarantees
will be made until version 1.0.0.

Example
=======

.. code-block:: python

    import dkpy
    import numpy as np

    # Load an example
    eg = dkpy.example_skogestad2006_p325()

    # Set up the D-K iteration method
    dk_iter = dkpy.DkIterListOrder(
        controller_synthesis=dkpy.HinfSynLmi(),
        structured_singular_value=dkpy.SsvLmiBisection(),
        d_scale_fit=dkpy.DScaleFitSlicot(),
        fit_orders=[4, 4, 4],
    )

    # Synthesize a controller
    omega = np.logspace(-3, 3, 61)
    block_structure = np.array([[1, 1], [1, 1], [2, 2]])
    K, N, mu, d_scale_fit_info, info = dk_iter.synthesize(
        eg["P"],
        eg["n_y"],
        eg["n_u"],
        omega,
        block_structure,
    )

Contributing
============

To install the pre-commit hook, run

.. code-block:: sh

   $ pip install -r requirements.txt
   $ pre-commit install

in the repository root.

Citation
========

If you use this software in your research, please cite it as below or see
``CITATION.cff``.

.. code-block:: bibtex

    @software{dahdah_dkpy_2024,
        title={{decargroup/dkpy}},
        url={https://github.com/decargroup/dkpy},
        author={Steven Dahdah and James Richard Forbes},
        version = {{v0.1.5}},
        year={2024},
    }
