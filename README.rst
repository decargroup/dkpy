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

    dk_iter = dkpy.DkIterFixedOrder(
        controller_synthesis=dkpy.HinfSynLmi(
            lmi_strictness=1e-7,
            solver_params=dict(
                solver="MOSEK",
                eps=1e-9,
            ),
        ),
        structured_singular_value=dkpy.SsvLmiBisection(
            bisection_atol=1e-5,
            bisection_rtol=1e-5,
            max_iterations=1000,
            lmi_strictness=1e-7,
            solver_params=dict(
                solver="MOSEK",
                eps=1e-9,
            ),
        ),
        transfer_function_fit=dkpy.TfFitSlicot(),
        n_iterations=3,
        fit_order=4,
    )

    omega = np.logspace(-3, 3, 61)
    block_structure = np.array([[1, 1], [1, 1], [2, 2]])
    K, N, mu, info = dk_iter.synthesize(P, n_y, n_u, omega, block_structure)

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
        version = {{v0.1.4}},
        year={2024},
    }
