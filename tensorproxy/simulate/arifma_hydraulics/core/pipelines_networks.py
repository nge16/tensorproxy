# """Функции для моделирования стационарных и нестационарных режимов течения
# газа в газотранспортных системах."""

# from typing import Callable, Tuple
# from functools import partial

# import jax
# from jax import jit
# import jax.typing as jtp
# import jax.numpy as jnp
# import jax.experimental.sparse as jsparse
# from jax import config

# from tensorproxy.utils.jax_utils import (
#     val_and_jacfwd,
#     stack_leaves,
#     unstack_leaves,
#     amax,
# )


# config.update("jax_enable_x64", True)


# # @jax.jit
# def _get_Λ_B(
#     edges_attr: jtp.DTypeLike,
#     nodes_attr: jtp.DTypeLike,
#     pipes_nodes: jtp.DTypeLike,
# ) -> jax.Array:
#     """Рассчитывает матрицы коэффициентов Λ и B.

#     Args:
#         edges_attr (jtp.ArrayLike): значения атрибутов дуг расчетного
#           графа ["L", "Di", "E", "k", "Tокр." "Qin", "Qout"], shape [n,]
#         nodes_attr (jtp.ArrayLike): тензор значений в узлах расчетного
#           графа Φ = {p, T, Tос, Q}, shape [m, |Φ|]
#         pipes_nodes (jtp.ArrayLike): массив кортежей
#             (узел начала газопровода, узел конца газопровода)

#     Returns:
#         jax.Array: константы, зависящие от параметров газопроводов
#     """

#     # TODO: реализовать уточнение плотности
#     ρ = 0.68

#     # плотность воздуха при стандартных условиях
#     ρ_air = 1.2046

#     # относительная плотность газа по воздуху
#     Δ = ρ / ρ_air

#     # газовая постоянная для воздуха, [R] = Дж/(кг*К)
#     R_air = 287.04

#     # [d] = м, [E] - безр., [Q] - млн м3/сут
#     # Din = jnp.array([attr["d"] for _, _, attr in edges_attr])
#     # E = jnp.array([attr.get("E", 0.95) for _, _, attr in edges_attr])
#     # K = jnp.array([attr.get("K", 0.03 * 1e-3) for _, _, attr in edges_attr])
#     # L = jnp.array([attr["l"] for _, _, attr in edges_attr])

#     L = edges_attr[..., :, 0]
#     Din = edges_attr[..., :, 1]
#     E = edges_attr[..., :, 2]
#     K = edges_attr[..., :, 3]
#     Te = edges_attr[..., :, 4]

#     # TODO: реализовать расчет динамической вязкости газа
#     # μ = 0.0

#     # NOTE: Re зависит от расхода газа по дугам => для учета расхода
#     # в трейсинге нужно использовать параметры qs, qf (аналогично
#     # nodes_attr = nodes_attr.at[unknown_p_nodes, 0].set(p))
#     # def Re(Q):
#     #     return 17750.0 * Q * Δ / (D * μ)

#     # учесть в расчетах 158 / Re
#     λ_avg = (0.067 / E**2) * (
#         # 158 / Re +
#         2
#         * K
#         / Din
#     ) ** 0.2

#     # TODO: реализовать расчет Z
#     z_avg = 0.92

#     def _t_avg(edge):
#         """Рассчитывает среднюю температуру газа в трубе."""
#         u, v = edge

#         # Реализовать расчет
#         return 300.0
#         T_env = (nodes_attr[u, 2] + nodes_attr[v, 2]) / 2.0
#         T_0 = nodes_attr[u, 1]
#         T_L = nodes_attr[v, 1]
#         T_avg = T_env + (T_0 - T_L) / jax.lax.log(
#             (T_0 - T_env) / (T_L - T_env)
#         )
#         return T_avg

#     # [T] = К
#     ftav = jax.vmap(_t_avg)
#     # T_avg = ftav(jnp.array([((u, v)) for u, v, _ in edges_attr]))
#     T_avg = ftav(jnp.array([(u, v) for u, v in pipes_nodes]))

#     # стр. 125 С-С, ф-ла 3.5
#     Λ = (
#         16.0
#         * λ_avg
#         * R_air
#         * (ρ_air**2)
#         * Δ
#         * z_avg
#         * T_avg
#         * L
#         / (jnp.pi**2 * Din**5)
#     )

#     B = 4.0 * R_air * ρ_air * T_avg * z_avg / (jnp.pi * Din**2 * L)

#     return Λ, B


# # @jax.jit
# def steady_residuals(x: jtp.ArrayLike, params: dict) -> jax.Array:
#     """Рассчитывает невязки системы уравнений, описывающих стационарное
#     течение газа в газотранспортной системе.

#     Args:
#         x (jtp.ArrayLike): вектор искомых переменных (в едином массиве):
#             - p (jtp.ArrayLike): вектор давлений газа в узлах
#             - q (jtp.ArrayLike): вектор расходов газа по дугам
#         params (dict): словарь, содержащий данные о задаче:
#             - At (jtp.ArrayLike): разреженная матрица инциденций
#                 (в формате sparse.BCOO)
#             - pipes_attr (jtp.DTypeLike): значения атрибутов дуг
#                 расчетного графа, shape [n,]
#             - nodes_attr (jtp.ArrayLike): тензор значений в узлах
#                 расчетного графа Φ = {p, T, Tос, Q}, shape [m, |Φ|]
#             - `unknown_p_nodes` (jtp.ArrayLike, dtype="int32"): узлы,
#                 в которых не задано давление газа
#             - `pipes_nodes` (jtp.ArrayLike): массив кортежей
#                 (узел начала газопровода, узел конца газопровода)

#     See also:
#         - Сухарев М.Г., Самойлов Р.В. стр. 105
#         - Попов Р.В. дисс-я, стр. 44

#     Returns:
#         jax.Array: невязки системы уравнений, описывающих стационарное
#         течение газа в ГТС
#     """
#     At: jsparse.BCOO = params["At"]
#     pipes_attr: jtp.DTypeLike = params["pipes_attr"]
#     nodes_attr: jtp.ArrayLike = params["nodes_attr"]
#     unknown_p_nodes: jtp.ArrayLike = params["unknown_p_nodes"]
#     known_p_nodes: jtp.ArrayLike = params["known_p_nodes"]
#     pipes_nodes: jtp.ArrayLike = params["pipes_nodes"]

#     n = At.shape[1]
#     k = len(unknown_p_nodes)
#     p, q = jnp.split(x, [k])

#     # для выполнения трейсинга параметров с целью
#     # отслеживания вычислений и дифференцирования функций
#     nodes_attr = nodes_attr.at[unknown_p_nodes, 0].set(p)

#     P1: jtp.ArrayLike = nodes_attr[unknown_p_nodes, 0] ** 2
#     P2: jtp.ArrayLike = nodes_attr[known_p_nodes, 0] ** 2
#     Q: jtp.ArrayLike = nodes_attr[unknown_p_nodes, 3]

#     # Убирает размерность батча для устранения ошибки при транспонировании
#     unbatch = lambda arr: jsparse.bcoo_update_layout(arr, n_batch=0)
#     A1, A2 = unbatch(At[..., unknown_p_nodes, :]), unbatch(
#         At[known_p_nodes, :]
#     )

#     αs, _ = _get_Λ_B(pipes_attr, nodes_attr, pipes_nodes)

#     diag_indices = jnp.array([[i, i] for i in range(n)])
#     mat_diag = lambda arr: jsparse.BCOO(
#         (arr, diag_indices),
#         shape=(n, n),
#     )

#     Λ = mat_diag(αs)
#     Θs = mat_diag(jnp.abs(q))

#     F0 = A1.T @ P1 - Λ @ Θs @ q + A2.T @ P2
#     F1 = A1 @ q - Q

#     return jnp.concat([F0, F1])


# # @jax.jit
# def unsteady_residuals(x: jtp.ArrayLike, params: dict, τ=60 * 5) -> jax.Array:
#     """Рассчитывает невязки системы обыкновенных дифференциальных уравнений,
#     описывающих нестационарное течение газа в газотранспортной системе.

#     Args:
#         x (jtp.ArrayLike): вектор искомых переменных (в едином массиве):
#             - p (jtp.ArrayLike): вектор давлений газа в узлах
#             - qs (jtp.ArrayLike): вектор начальных расходов газа по дугам
#             - qf (jtp.ArrayLike): вектор конечных расходов газа по дугам
#         params (dict): словарь, содержащий данные о задаче:
#             - At (jtp.ArrayLike): разреженная матрица инциденций
#                 (в формате sparse.BCOO)
#             - edges_attr (jtp.ArrayLike): значения атрибутов дуг расчетного
#               графа, shape [n,]
#             - nodes_attr (jtp.ArrayLike): тензор значений в узлах расчетного
#               графа Φ = {p, T, Tос, Q}, shape [m, |Φ|]
#             - `pj_prev` (jtp.ArrayLike): массив значений давления газа в узлах
#             расчетного графа на предыдущем временном слое
#             - `qs_prev` (jtp.ArrayLike): массив значений расхода газа в
#               началах дуг на предыдущем временном слое
#             - `qf_prev` (jtp.ArrayLike): массив значений расхода газа в конце
#                 дуг на предыдущем временном слое
#             - `unknown_p_nodes` (jtp.ArrayLike, dtype="int32"): узлы, в которых
#                 не задано давление газа
#         τ (int): шаг шкалы разбиения рассматриваемого отрезка времени
#             [0, t_end] равномерной шкалой time_index * τ, time_index = 0,
#             1, ..., K, где τ = t_end / K

#     Note:
#       Используются следующие обозначения:
#         At (chex.Array): матрица инциденций графа Gt, shape [m x n]
#         A_tr(chex.Array): матрица, ограниченная узлами с незаданными давлениями
#         As (chex.Array): матрица, содержащая элементы матрицы At, равные +1
#         Af (chex.Array): матрица, содержащая элементы матрицы At, равные -1
#         n (int): количество дуг графа Gt
#         m (int): количество вершин графа Gt
#         P (chex.Array): вектор потенциалов ||p_i^2||, shape [m x 1]
#         k: эквивалетная шероховатость труб
#         ρ: плотность газа при с.у., кг/м3
#         Θs: диагональная матрица, на главной диагонали которой
#             располагаются модули расходов газа на входе r-й дуги,
#             shape [n x n]
#         Θf: диагональная матрица, на главной диагонали которой
#             располагаются модули расходов газа на выходе r-й дуги,
#             shape [n x n]
#         Λ: диагональная матрица, на главной диагонали которой
#             располагаются коэффициенты, shape [n x n]
#         B: диагональная матрица, на главной диагонали которой
#             располагаются коэффициенты, shape [n x n]
#         Ψs: диагональная матрица, shape [n x n]
#         Ψf: диагональная матрица, shape [n x n]
#         Ωs: диагональная матрица, shape [n x n]
#         Ωf: диагональная матрица, shape [n x n]

#     Returns:
#         jax.Array: невязки системы уравнений, описывающих нестационарное
#           течение газа в ГТС
#     """

#     At: jsparse.BCOO = params["At"]
#     edges_attr: jtp.DTypeLike = params["edges_attr"]
#     nodes_attr: jtp.ArrayLike = params["nodes_attr"]

#     start_nodes = [u for u, _, _ in edges_attr]
#     end_nodes = [v for _, v, _ in edges_attr]

#     pj_prev: jtp.ArrayLike = params["pj_prev"]
#     qs_prev: jtp.ArrayLike = params["qs_prev"]
#     qf_prev: jtp.ArrayLike = params["qf_prev"]
#     unknown_p_nodes: jtp.ArrayLike = params["unknown_p_nodes"]

#     n = At.shape[1]
#     A_tr = At[unknown_p_nodes, :]

#     # убирает размерность батча для устранения ошибки при транспонировании
#     unbatch = lambda arr: jsparse.bcoo_update_layout(arr, n_batch=0)
#     As = unbatch(
#         jsparse.BCOO(
#             (A_tr.data * (A_tr.data > 0), A_tr.indices), shape=A_tr.shape
#         )
#     )
#     Af = unbatch(
#         jsparse.BCOO(
#             (A_tr.data * (A_tr.data < 0), A_tr.indices), shape=A_tr.shape
#         )
#     )

#     # размерность по каждой переменной
#     ind = [
#         len(params["unknown_p_nodes"]),
#         len(params["unknown_p_nodes"]) + len(params["edges_attr"]),
#     ]
#     pj, qs, qf = jnp.split(x, ind)

#     # трейсинга параметров для дифференцирования функций
#     nodes_attr = nodes_attr.at[unknown_p_nodes, 0].set(pj)

#     Q: jtp.ArrayLike = nodes_attr[unknown_p_nodes, 3]
#     P: jtp.ArrayLike = nodes_attr[:, 0] ** 2

#     αs, βs = _get_Λ_B(edges_attr, nodes_attr)

#     diag_indices = jnp.array([[i, i] for i in range(n)])
#     mat_diag = lambda arr: jsparse.BCOO(
#         (arr, diag_indices),
#         shape=(n, n),
#     )

#     Λ = mat_diag(αs)
#     B = mat_diag(βs)
#     Binv = mat_diag(1 / βs)
#     Θs = mat_diag(jnp.abs(qs))
#     Θf = mat_diag(jnp.abs(qf))
#     Ψs = mat_diag(qs * jnp.abs(qs) / (nodes_attr[start_nodes, 0] ** 2))
#     Ψf = mat_diag(qf * jnp.abs(qf) / (nodes_attr[end_nodes, 0] ** 2))
#     Ωs = mat_diag(2 * jnp.abs(qs) / nodes_attr[start_nodes, 0])
#     Ωf = mat_diag(2 * jnp.abs(qf) / nodes_attr[end_nodes, 0])

#     a = 14.0  # 12, или 14 - 16 (стр. 30 дисс. Попова)
#     F0 = (
#         At.T @ P
#         - 0.5 * Λ @ (Θs @ qs + Θf @ qf)
#         + 2.0
#         / (a * τ)
#         * Λ
#         @ Binv
#         @ (Θs @ As.T @ (pj - pj_prev) + Θf @ Af.T @ (pj - pj_prev))
#     )

#     F1 = (
#         (As.T - Af.T) @ (pj - pj_prev)
#         + 1.0
#         / a
#         * Λ
#         @ (
#             Ψs @ As.T @ (pj - pj_prev)
#             - Ωs @ (qs - qs_prev)
#             + Ψf @ Af.T @ (pj - pj_prev)
#             - Ωf @ (qf - qf_prev)
#         )
#         - 2 * τ * B @ (qs - qf)
#     )

#     F2 = As @ qs + Af @ qf - Q

#     return jnp.concat([F0, F1, F2])


# # @partial(jit, static_argnums=(0))
# def loss(
#     func: Callable[..., jax.Array],
#     x: jtp.ArrayLike,
#     params: jtp.DTypeLike,
# ) -> jax.Array:
#     return jnp.linalg.norm(func(x, params))


# # @partial(jit, static_argnums=(0))
# def callback_func(
#     self,
#     cnt: int,
#     err: float = None,
#     *args,
#     fev=None,
#     ltime=None,
#     verbose: bool = True,
# ) -> None:
#     """Выводит информацию об очередной итерации.

#     Args:
#         cnt (_type_): счетчик итераций
#         err (_type_): ошибка на итерации
#         fev (_type_, optional): _description_. Defaults to None.
#         ltime (_type_, optional): _description_. Defaults to None.
#         verbose (bool, optional): выводить ли информацию
#     """
#     msg = f"    Итерация {cnt}"
#     if verbose:
#         jax.debug.print(msg)


# # @jax.jit
# def _cond_func(carry: tuple) -> bool:
#     (xi, params, eps, cnt), (func, loss_func, verbose, maxit, tol) = carry
#     cond = cnt < maxit
#     cond = jnp.logical_and(cond, eps > tol)
#     cond = jnp.logical_and(cond, ~jnp.isnan(eps))
#     verbose = jnp.logical_and(cnt, verbose)
#     # jax.debug.callback(self.callback_func, cnt, eps, verbose=verbose)
#     return cond


# # @jax.jit
# def _body_func(carry: tuple):
#     (xi, params, eps, cnt), (func, loss_func, verbose, maxit, tol) = carry
#     xi_old = xi
#     f, jac = val_and_jacfwd(func, argnums=(0))(xi, params)
#     xi -= jax.scipy.linalg.solve(jac, f)
#     eps = amax(xi - xi_old)
#     # eps = loss_func(func, xi, params)
#     return (xi, params, eps, cnt + 1), (func, loss_func, verbose, maxit, tol)


# # @jax.jit
# def solve(
#     func: Callable,
#     x0: jtp.ArrayLike,
#     params: jtp.DTypeLike,
#     loss_func: Callable = None,
#     tol: float = 1.0e-1,
#     maxit: int = 50,
#     verbose: bool = True,
# ) -> Tuple[jtp.ArrayLike, float]:
#     """Решает задачу потокораспределения методом Ньютона.

#     Args:
#         func (Callable): функция, возвращающая невязки уравнений
#         x0 (jtp.ArrayLike): начальное приближение
#         params (jtp.DTypeLike): параметры, описывающие расчетный граф
#         loss_func (Callable): задел на будущее для использования
#             произвольной функции loss при принятии решений
#             об остановке вычислительной процедуры
#         tol (float, optional): максимальная допустимая ошибка.
#           По умолчанию 1.0e-1.
#         maxit (int, optional): максимальное число итераций.
#           По умолчанию 50.
#         verbose (bool, optional): выводить ли промежуточную информацию

#     Returns:
#         Tuple[jtp.ArrayLike, float]: найденное значение вектора x, ошибка
#     """
#     (xi, params, eps, cnt), _ = jax.lax.while_loop(
#         _cond_func,
#         _body_func,
#         ((x0, params, 1.0, 0), (func, loss_func, verbose, maxit, tol)),
#     )
#     return xi, eps


# # @jax.jit
# # def solve_unsteady_jit(
# #     x0: jtp.ArrayLike,
# #     params: jtp.ArrayLike,
# #     tol: float = 1.0e-1,
# #     maxit: int = 50,
# #     timesteps: int | None = None,
# #     verbose: bool = True,
# # ):
# #     """Выполняет расчет нестационарного режима ГТС.

# #     Компилируемая версия функции для расчета нестационарного режима.

# #     Args:
# #         x0 (jtp.ArrayLike): начальное приближение для расчета стационарного
# #           режима
# #         params (jtp.DTypeLike): параметры, описывающие расчетный граф
# #         tol (float, optional): максимальная допустимая ошибка.
# #           По умолчанию 1.0e-1.
# #         maxit (int, optional): максимальное число итераций.
# #           По умолчанию 50.
# #         timesteps (int | None, optional): максимальное число рассчитываемых
# #           слоев по времени
# #         verbose (bool, optional): выводить ли промежуточную информацию

# #     Returns:
# #         _type_: _description_
# #     """
# #     _partial = lambda f: jax.tree_util.Partial(f)

# #     unstacked_params = unstack_leaves(params)
# #     # Расчет стационарного режима
# #     xstdy, eps = solve(
# #         _partial(steady_residuals),
# #         x0,
# #         unstacked_params[0],
# #         maxit=maxit,
# #         tol=tol,
# #         verbose=verbose,
# #     )
# #     p, q = jnp.split(xstdy, [len(unstacked_params[0]["unknown_p_nodes"])])

# #     # Расчет нестационарного режима
# #     xns0 = jnp.concat([p.copy(), q.copy(), q.copy()])

# #     @jax.jit
# #     def _sscan(carry, params):
# #         xprev, (maxit, tol, verbose) = carry

# #         ind = [
# #             len(params["unknown_p_nodes"]),
# #             len(params["unknown_p_nodes"]) + len(params["edges_attr"]),
# #         ]
# #         p_prev, qs_prev, qf_prev = jnp.split(xprev, ind)

# #         params["pj_prev"] = p_prev.copy()
# #         params["qs_prev"] = qs_prev.copy()
# #         params["qf_prev"] = qf_prev.copy()

# #         xi0 = jnp.concat(
# #             [
# #                 params["pj_prev"].copy(),
# #                 params["qs_prev"].copy(),
# #                 params["qf_prev"].copy(),
# #             ]
# #         )

# #         xopt, eps = solve(
# #             _partial(unsteady_residuals),
# #             xi0,
# #             params,
# #             maxit=maxit,
# #             tol=tol,
# #             verbose=verbose,
# #         )
# #         return (xopt, (maxit, tol, verbose)), (xopt)

# #     res = jax.lax.scan(
# #         _sscan,
# #         (xns0, (maxit, tol, verbose)),
# #         stack_leaves(unstacked_params[1:]),
# #     )

# #     return res


# def solve_steady_state(
#     x0: jtp.ArrayLike,
#     params: jtp.DTypeLike,
#     tol: float = 1.0e-1,
#     max_iter: int = 50,
#     verbose: int = 1,
# ):
#     """Решает задачу моделирования стационарного режима работы ГТС."""
#     xopt, eps = solve(
#         (jax.tree_util.Partial(steady_residuals)),
#         x0,
#         params,
#         maxit=max_iter,
#         tol=tol,
#         verbose=verbose,
#     )

#     return xopt, eps


# def solve_unsteady_iterable(  # noqa: C901
#     timesteps: int,
#     tol: float = 1.0e-1,
#     max_iter: int = 50,
#     verbose: int = 1,
# ) -> None:
#     """Решает задачу моделирования нестационарного режима работы ГТС.

#     Note:
#         Решает задачу послойно во временным шагам.

#     Args:
#         timesteps (int): число шагов симуляции нестационарного режима
#         tol (float, optional): точность решения
#         max_iter (int, optional): максимальное число итераций
#         timesteps (int, optional): ограничение на число шагов
#     """
#     T = timesteps
#     # solved_timesteps = 0

#     if verbose > 0:
#         print(f"Период прогнозирования нестационарного режима: {T} шагов")

#     for t in range(T):
#         if verbose > 0:
#             print(f"\tСлой {t}:")

#         unsteady = t > 0
#         x0, params = self._problem(t, unsteady)

#         _partial = lambda f: jax.tree_util.Partial(f)
#         xopt, eps = jhydro.solve(
#             (
#                 _partial(jhydro.steady_residuals)
#                 if not unsteady
#                 else _partial(jhydro.unsteady_residuals)
#             ),
#             x0,
#             params,
#             maxit=max_iter,
#             tol=tol,
#             verbose=verbose,
#         )

#         if eps > tol:
#             raise Exception(f"Не удалось найти решение на слое {t}")

#         if verbose > 0:
#             print(f"\tРешение на слое {t=}: {xopt}\n")

#         self.solved_timesteps = t + 1
#         gt = self.model.snapshots[t]
#         unk = params["unknown_p_nodes"].tolist()

#         if unsteady:
#             shapes = np.array(
#                 [
#                     params["pj_prev"].shape[-1],
#                     params["qs_prev"].shape[-1],
#                     params["qf_prev"].shape[-1],
#                 ]
#             )

#             xopt = np.array(xopt.tolist())
#             pj, qs, qf = np.split(xopt, np.cumsum(shapes)[:-1])

#             for i, node in enumerate(unk):
#                 gt.nodes[node]["p"] = float(pj[i])

#             for j, (u, v, key) in enumerate(gt.edges(keys=True)):
#                 gt[u][v][key]["qs"] = float(qs[j])
#                 gt[u][v][key]["qf"] = float(qf[j])

#         else:
#             p, q = jnp.split(xopt, [len(unk)])
#             p, q = p.tolist(), q.tolist()

#             for i, node in enumerate(unk):
#                 gt.nodes[node]["p"] = p[i]

#             for j, (u, v, key) in enumerate(gt.edges(keys=True)):
#                 gt[u][v][key]["qs"] = gt[u][v][key]["qf"] = float(q[j])
