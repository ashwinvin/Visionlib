import setuptools

with open("README.md", "r") as rd:
    long_description = rd.read()

with open('requirements.txt', 'r') as rq:
    requirements = rq.read().strip().split('\n')

setuptools.setup(
    name="visionlib",
    version="1.4.5",
    author="Ashwin Vinod",
    author_email="ashwinvinodsa@gmail.com",
    url="https://github.com/ashwinvin/Visionlib",
    download_url="https://github.com/ashwinvin/Visionlib/archive/v1.4.5.tar.gz",
    description="    ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Deep learning", "Vision", "cv"],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.6",
)
