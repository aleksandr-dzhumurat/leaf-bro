from setuptools import find_packages, setup

setup(
    name="leaf_bro",
    packages=find_packages(exclude=["leaf_bro_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        # custom
        "google-cloud-storage",
        "gcsfs"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
