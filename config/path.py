from os import sep
from os.path import join, dirname, realpath


PROJECT_ROOT = join(sep, *dirname(realpath(__file__)).split(sep)[: -1])

DATA_PATH = join(
    PROJECT_ROOT, "data", "rubber", "rubber_data_2021", "*", "*.tif")
CSV_PATH = join(
    PROJECT_ROOT, "data", "rubber", "rubber_data_2021", "func_data_norm.csv")
