import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="yolo_service", # Replace with your own username
    version="0.0.1",
    author="Luis Restrepo",
    author_email="luisrestrepo1995@gmail.com",
    description="Package for YOLO service",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)