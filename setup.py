import setuptools

with open("README.md", "r") as rd:
    long_description = rd.read()

setuptools.setup(
    name="tunnel_vision",
    version="0.6",
    author="Ashwin Vinod",
    author_email="ashwinvinodsa@gmail.com",
    description="This library will make your life much easier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
