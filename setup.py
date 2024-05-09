from setuptools import setup

setup(
    name="hd_mock_data",
    version="1.0",
    description="Mock forecasting data for CMB-HD",
    url="https://github.com/CMB-HD/hdMockData",
    author="CMB-HD Collaboration",
    python_requires=">=3",
    #install_requires=["numpy"],
    packages=["hd_mock_data"],
    include_package_data=True,
)
