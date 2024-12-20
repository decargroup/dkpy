D-K iteration methods
=====================

The D-K iteration methods provided by ``dkpy`` are presented below. Each one
implements the interface specified in :class:`DkIteration`. The difference
between these methods is the way the D-scale fit order is selected. It can
either be fixed, specified via a list, selected automatically, or selected
interactively.

.. autosummary::
   :toctree: _autosummary/

   dkpy.DkIterFixedOrder
   dkpy.DkIterListOrder
   dkpy.DkIterAutoOrder
   dkpy.DkIterInteractiveOrder

Each :func:`DkIteration.synthesize` method returns (among other things) a list
of :class:`IterResult` objects. These objects summarize the status of the D-K
iteration process at each step. They can be plotted with :func:`plot_D` and
:func:`plot_mu` to assess the accuracy of the D-scale fit and its impact on the
structured singular value.

.. autosummary::
   :toctree: _autosummary/

   dkpy.IterResult
   dkpy.plot_mu
   dkpy.plot_D

Controller synthesis
====================

Supported continuous-time H-infinity controller synthesis methods are provided
below. Each one implements the interface specified in
:class:`ControllerSynthesis`.

.. autosummary::
   :toctree: _autosummary/

   dkpy.HinfSynSlicot
   dkpy.HinfSynLmi
   dkpy.HinfSynLmiBisection


Structured singular value
=========================

Supported structured singular value computation methods are provided below.
Only one approach is provided, which implements the interface in
:class:`StructuredSingularValue`. The LMI solver settings may need to be
adjusted depending on the problem.

.. autosummary::
   :toctree: _autosummary/

   dkpy.SsvLmiBisection

D-scale fit
===========

Supported D-scale fitting methods are provided below. Only one approach is
provided currently, which implements the interface in :class:`DScaleFit`. There
are currently no ways to customize the D-scale magnitude fitting process beyond
selecting the order in :func:`DScaleFit.fit`.

.. autosummary::
   :toctree: _autosummary/

   dkpy.DScaleFitSlicot

Extending ``dkpy``
==================

The abstract classes defining the structure of ``dkpy`` are presented below.
Anyone aiming to extend or customize ``dkpy`` should familiarize themselves
with them.

.. autosummary::
   :toctree: _autosummary/

   dkpy.DkIteration
   dkpy.ControllerSynthesis
   dkpy.StructuredSingularValue
   dkpy.DScaleFit
