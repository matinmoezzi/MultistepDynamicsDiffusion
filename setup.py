from setuptools import setup, find_packages

setup(
    name="multistep_dynamicsdiffusion",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here, e.g.
        # "numpy>=1.0",
        # "pandas>=1.0",
    ],
    entry_points={},
    python_requires=">=3.9",
    # Add metadata about your project
    author="Matin Moezzi",
    author_email="matin.moezzi@mail.utoronto.ca",
    description="Model-based Reinforcement Learning with Diffusion Models",
    # long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    # license="MIT",
    url="https://github.com/matinmoezzi/MultistepDynamicsDiffusion",
)
