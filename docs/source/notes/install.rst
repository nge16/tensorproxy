Установка
=========

Зависимости
------------

pyhydrosim имеет следующие **обязательные** зависимости:

* `Python <https://www.python.org/>`__ 3.10 или более позднюю версию,
* `pagmo C++ library <https://esa.github.io/pagmo2/>`__, версии 2.19 и позже,
* `NumPy <https://numpy.org/>`__,

Также pyhydrosim имеет следующие **опциональные** runtime-зависимости:

* `Matplotlib <https://matplotlib.org/>`__, для отрисовки 
* `NetworkX <https://networkx.github.io/>`__, для оперирования с графами
* `SciPy <https://www.scipy.org/>`__, который используется в :class:`~pyhydrosim.optimize.solvers.scipy_solver`.