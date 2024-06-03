import setuptools as st
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).absolute().parent

packages = st.find_packages()
print(packages)

st.setup(
    name="marglicot",
    version="1.0",
    py_modules=packages,
    package_dir={k: v for k, v in packages},
)