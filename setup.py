import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

reqs = ['torch', 'torchvision']
setuptools.setup(
    name="rosenbrock",
    version="0.1",
    author="Ali Siahkoohi",
    author_email="alisk@gatech.edu",
    description="Rosenbrock distribution, including unnormalized density and analytic sampler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alisiahkoohi/rosenbrock",
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages()
)
