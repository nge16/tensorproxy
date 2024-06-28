import numpy as np
from scipy.sparse import csr_array


def csr_vappend(a: csr_array, b: csr_array) -> csr_array:
    """Добавляет вторую разреженную матрицу к первой.

      Перезаписывает первую матрицу. Копирует data, indices, indptr.

    Args:
        a (csr_array): разреженная матрица, к которой добавляются значения
        b (csr_array): разреженная матрица, которая добавляется
          к первой матрице

    Returns:
        csr_array: новая разреженная матрица
    """
    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a
