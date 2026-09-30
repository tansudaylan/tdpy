from setuptools import setup, find_packages

setup(
    name = 'tdpy',
    packages = find_packages(),
    version = '1.2',
    description = 'A python library of numberical routines', \
    author = 'Tansu Daylan',
    author_email = 'tansu.daylan@gmail.com',
    url = 'https://github.com/tansudaylan/tdpy',
    download_url = 'https://github.com/tansudaylan/tdpy',
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python'],
    install_requires=['numpy', 'matplotlib', 'astropy', 'scipy', 'pandas', 'Pillow', 'scikit-learn', 'tesswcs', 'tqdm'],
    python_requires='>=3'
    )

