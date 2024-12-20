Examples
========

In all the examples on this page, three iterations with 4th order D-scale fits
are used to reproduce the example from [SP06]_, Table 8.2 (p. 325). Each example
recovers the same result, but with a different way to specify the D-scale fit
orders.

This example is quite numerically challenging, so if you encounter a solver
error, you may need to experiment with solver tolerances.

D-K iteration with fixed fit order
----------------------------------

In this example, the number of iterations is fixed to 3 and the order is fixed
to 4.

.. literalinclude:: ../examples/1_example_dk_iter_fixed_order.py
  :language: python

Output::

    mu=1.0132789611816406

.. image:: _static/plot_mu.png
.. image:: _static/plot_D.png

D-K iteration with list of fit orders
-------------------------------------

In this example, the orders are specified in a list. They can be specified
manually for each entry of the D matrix, or they can all be set to the same
integer.

.. literalinclude:: ../examples/2_example_dk_iter_list_order.py
  :language: python

Output::

    mu=1.0132789611816406

.. image:: _static/plot_mu.png
.. image:: _static/plot_D.png

D-K iteration with automatically selected fit orders
----------------------------------------------------

In this example, multiple fit orders are attempted up to a maximum, and the one
with the lowest relative error is selected.

.. literalinclude:: ../examples/3_example_dk_iter_auto_order.py
  :language: python

Output::

    INFO:DkIterAutoOrder:Iteration: 0, mu: 1.1792325973510742
    INFO:DkIterAutoOrder:Order 0 relative error: 0.5122457769147215
    INFO:DkIterAutoOrder:Order 1 relative error: 0.34431183690576633
    INFO:DkIterAutoOrder:Order 2 relative error: 0.8970100659296376
    INFO:DkIterAutoOrder:Order 3 relative error: 0.030844892155775263
    INFO:DkIterAutoOrder:Order 4 relative error: 0.015896380940858944
    INFO:DkIterAutoOrder:Reached max fit order, selecting order 4
    INFO:DkIterAutoOrder:Iteration: 1, mu: 1.0256481170654297
    INFO:DkIterAutoOrder:Order 0 relative error: 9.350642790785383
    INFO:DkIterAutoOrder:Order 1 relative error: 1.327796980729705
    INFO:DkIterAutoOrder:Order 2 relative error: 7.58063969442474
    INFO:DkIterAutoOrder:Order 3 relative error: 0.13472058625314948
    INFO:DkIterAutoOrder:Order 4 relative error: 0.05262584627773765
    INFO:DkIterAutoOrder:Reached max fit order, selecting order 4
    INFO:DkIterAutoOrder:Iteration: 2, mu: 1.0201168060302734
    INFO:DkIterAutoOrder:Order 0 relative error: 28.96943897648902
    INFO:DkIterAutoOrder:Order 1 relative error: 3.9918344794684804
    INFO:DkIterAutoOrder:Order 2 relative error: 7.000978128580445
    INFO:DkIterAutoOrder:Order 3 relative error: 0.22159755671770862
    INFO:DkIterAutoOrder:Order 4 relative error: 0.06377761160336083
    INFO:DkIterAutoOrder:Reached max fit order, selecting order 4
    INFO:DkIterAutoOrder:Iteration: 3, mu: 1.0132789611816406
    INFO:DkIterAutoOrder:Iteration terminated: reached maximum number of iterations
    INFO:DkIterAutoOrder:Iteration complete
    mu=1.0132789611816406

.. image:: _static/plot_mu.png
.. image:: _static/plot_D.png

D-K iteration with interactively selected fit orders
----------------------------------------------------

In this example, the user is prompted to select a D-scale fit order at each
iteration. The user is shown the frequency-by-frequency and fit structured
singular value plots at each iteration.

.. literalinclude:: ../examples/4_example_dk_iter_interactive.py
  :language: python

.. image:: _static/interactive_1.png

Prompt::

    Close plot to continue...
    Select order (<Enter> to end iteration): 4

.. image:: _static/interactive_2.png

Prompt::

    Close plot to continue...
    Select order (<Enter> to end iteration): 4

.. image:: _static/interactive_3.png

Prompt::

    Close plot to continue...
    Select order (<Enter> to end iteration): 4

.. image:: _static/interactive_4.png

Output::

    Close plot to continue...
    Select order (<Enter> to end iteration): 
    Iteration ended.
    mu=1.0132789611816406

.. image:: _static/plot_mu.png
.. image:: _static/plot_D.png

D-K iteration with a custom fit order selection method
------------------------------------------------------

In this example, a custom D-K iteration class is used to stop the iteration
after 3 iterations of 4th order fits.

.. literalinclude:: ../examples/5_example_dk_iteration_custom.py
  :language: python

Output::

    Iteration 0 with mu of 1.1792325973510742
    Iteration 1 with mu of 1.0256481170654297
    Iteration 2 with mu of 1.0201168060302734
    Iteration 3 with mu of 1.0132789611816406
    mu=1.0132789611816406

.. image:: _static/plot_mu.png
.. image:: _static/plot_D.png
