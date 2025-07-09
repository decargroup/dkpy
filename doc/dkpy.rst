``dkpy`` architecture overview
==============================

``dkpy`` implements each step in the D-K iteration algorithm as an abstract
class. This ensures that each step satisfies the interface required by the
algorithm while leaving the numerical implementation up to the user. Sensible
implementations of each of these steps are provided in ``dkpy``. However, the
user can create custom implementations via the abstract classes. Therefore,
anyone aiming to extend or customize ``dkpy`` should familiarize themselves
with them. The abstract classes are provided below.

.. autosummary::
   :toctree: _autosummary/

   dkpy.ControllerSynthesis
   dkpy.StructuredSingularValue
   dkpy.DScaleFit
   dkpy.DkIteration

The steps of the D-K iteration algorithm are as follows [SP06]_. 
1. Controller synthesis (:class:`ControllerSynthesis`): Synthesize an
   H-infinity controller for the scaled problem with fixed fitted D-scales.
2. Structured singular value and D-scale computation
   (:class:`StructuredSingularValue`): Compute the structured singular value
   and the D-scales over a discrete grid of frequencies with a fixed
   controller.
3. D-scale fit (:class:`DScaleFit`): Fit the magnitude of each D-scale to a
   stable minimum-phase LTI system.

The D-K iteration algorithm, specified by a :class:`DkIteration` object, loops
through these three steps until the robustness criteria are satisfied.
Therefore, the algorithm is specified using implementations of each of the
abstract classes :class:`ControllerSynthesis`,
:class:`StructuredSingularValue`, and :class:`DScaleFit`.

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

Uncertainty block structure
===========================

The uncertainty block structure is specified via an
:class:`UncertaintyBlockStructure` object, which encodes the block diagonal
uncertainty structure. The :class:`UncertaintyBlockStructure` object is
composed of individual uncertainty blocks that satisfy the interface in
:class:`UncertaintyBlock`, which are provided below.

.. autosummary::
   :toctree: _autosummary/

   dkpy.RealDiagonalBlock
   dkpy.ComplexDiagonalBlock
   dkpy.ComplexFullBlock

The :class:`UncertaintyBlockStructure` object can also be specified using the
MATLAB two-column array format (see `MATLAB documentation
<https://www.mathworks.com/help/robust/ref/mussv.html>`) for users that are
more comfortable with this notation.
