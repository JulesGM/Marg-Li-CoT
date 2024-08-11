import setuptools as st

modules = st.find_packages() 
print(modules)

st.setup(
    name="marglicot",
    version="1.0",
    py_modules=modules,
)