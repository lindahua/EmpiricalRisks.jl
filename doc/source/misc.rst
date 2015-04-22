Auxiliary Utilities
=====================

The package also provides other convenience tools.

.. function:: no_op(...)

    This function accepts arbitrary arguments and returns nothing.

    It is mainly used as a callback function where you don't really need to callback to do anything.


.. function:: shrink(x, t)

    Compute the following function:

    .. math::

        (x, t) \mapsto \begin{cases}
            x - t & (x > t) \\
            0 & (|x| \le t) \\
            x + t & (x < -t)
        \end{cases}

    Here, ``x`` can be either a scalar or an array (for vectorized computation).
