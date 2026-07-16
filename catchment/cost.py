from numba import njit
import numpy as np
from scipy.sparse import csr_matrix


def _create_cost_graph(dem, walls=None, enforce_uphill=False, enforce_downhill=False):
    data, ids = _create_graph_data_numba(
        dem.data,
        walls,
        enforce_uphill,
        enforce_downhill,
        cell_size=dem.rio.resolution()[0],
    )
    graph = csr_matrix(data, shape=(ids.size, ids.size))
    return graph


@njit
def _create_graph_data_numba(dem, walls, enforce_uphill, enforce_downhill, cell_size):
    nrows, ncols = dem.shape
    ids = np.arange(dem.size).reshape(dem.shape)
    row_inds = []
    col_inds = []
    data = []
    for row in range(nrows):
        for col in range(ncols):
            start = ids[row, col]

            if walls is not None:
                if walls[row, col]:
                    continue

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = row + dx
                    ny = col + dy
                    end = ids[nx, ny]

                    if walls is not None:
                        if walls[nx, ny]:
                            continue

                    if 0 <= nx < nrows and 0 <= ny < ncols:
                        cost = 1 if dx == 0 or dy == 0 else 1.41
                        delta_elevation = dem[nx, ny] - dem[row, col]

                        if delta_elevation < 0 and enforce_uphill:
                            continue

                        if delta_elevation > 0 and enforce_downhill:
                            continue

                        cost = cost * cell_size + np.abs(delta_elevation) + 1e-6

                        data.append(np.abs(cost))
                        row_inds.append(start)
                        col_inds.append(end)

    return (data, (row_inds, col_inds)), ids
