.. role:: class(code)

dkpy
====

.. image:: https://github.com/decargroup/dkpy/actions/workflows/test-package.yml/badge.svg
    :target: https://github.com/decargroup/dkpy/actions/workflows/test-package.yml
    :alt: Test package
.. image:: https://readthedocs.org/projects/dkpy/badge/?version=stable
    :target: https://dkpy.readthedocs.io/en/stable/?badge=stable
    :alt: Documentation status
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14511244.svg
    :target: https://doi.org/10.5281/zenodo.14511244
    :alt: DOI

``dkpy`` is a `D-K iteration <https://doi.org/10.1109/ACC.1994.735077>`_
library written in Python, aiming to build upon
`python-control <https://github.com/python-control/python-control>`_.

The package is currently a work-in-progress, and no API stability guarantees
will be made until version 1.0.0.

D-K iteration
=============

The standard robust control problem has the form::

              ┌─────────┐          
              │         │          
      w2 ┌────┤    Δ    │◄───┐ z2  
         │    │         │    │     
         │    └─────────┘    │     
         │    ┌─────────┐    │     
         └───►│         ├────┘     
    w1 ──────►│    P    ├──────► z1
         ┌───►│         ├────┐     
         │    └─────────┘    │     
         │    ┌─────────┐    │     
         │    │         │    │     
       u └────┤    K    │◄───┘ y   
              │         │          
              └─────────┘          

where ``P`` is the generalized plant, ``K`` is the controller, and ``Δ`` is an
uncertain LTI system whose H-infinity norm is less than or equal to 1.

Synthesizing a controller that makes the transfer matrix from ``w1`` to ``z1``
have an H-infinity norm less than 1 guarantees robust stability by the small
gain theorem.

When ``Δ`` has structure (*e.g.*, ``Δ = diag(Δ1, Δ2)``), this approach is too
conservative. Robust stability can instead be achieved by synthesizing a
controller whose **structured singular value**, ``µ``, is less than 1. Robust
performance problems can also be viewed as robust stability problems with
structured uncertainty.

Minimizing ``µ`` is much more challenging than minimizing the H-infinity norm.
D-K iteration is one method to do so. It relies on the fact that an upper bound
for ``µ`` is::

    µ(M) ≤ min σ̅(DMD⁻¹)
            D

where ``D`` is a complex matrix whose structure commutes with ``Δ``. More
specifically, for each full block in ``Δ``, the corresponding entry of ``D`` is
``d I``, where ``d`` is a scalar and ``I`` is the identity matrix. If ``Δ`` has
any entries of the form ``δ I``, then ``D`` has a full block in the
corresponding entry.

D-K iteration has the following steps, where ``D`` is initially identity.

#. Augment ``P`` with ``D`` and ``D⁻¹``, then synthesize an H-infinity controller.
#. Compute ``µ`` and ``D`` for the closed-loop system without the D-scalings
   over a range of discrete frequencies.
#. Fit a transfer matrix to ``D`` and repeat. Stop when ``µ < 1``.

The D-K iteration process is represented by :class:`dkpy.DkIteration`. The
steps of the process are represented by

#. :class:`dkpy.ControllerSynthesis`,
#. :class:`dkpy.StructuredSingularValue`, and
#. :class:`dkpy.DScaleFit`.

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
        doi={10.5281/zenodo.14511244},
        url={https://github.com/decargroup/dkpy},
        publisher={Zenodo},
        author={Steven Dahdah and James Richard Forbes},
        version = {{v0.1.8}},
        year={2024},
    }
