from setuptools import find_packages, setup

setup(
    name="leaf_bro",
    packages=find_packages(exclude=["leaf_bro_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        # custom
        "google-cloud-storage",
        "gcsfs",
        "tqdm",
        "requests",
        "elasticsearch",
        "openai",
        "tiktoken"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
