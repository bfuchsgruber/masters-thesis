

def calc_rmse(one, two) -> float:
    rmse = ((one - two) ** 2).mean() ** .5
    return rmse