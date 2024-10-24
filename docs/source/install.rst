Installation
============
*Spring* is provided as a python package. However, under the hood, its implementation relies on C++ and CUDA, thus requiring a compilation step.
On the hardware side, a CUDA-capable GPU is required (i.e. NVidia GPU).


On the software side, the following tools are necessary to compile from source:
    - C++ compiler (compatible with the C++17 standard)
    - CUDA compiler (CUDA 11 or above)
    - cmake (3.17 or above)
    
At the moment, no pre-compiled packages are available, and the software has to be built from source. However, the complete installation process is based on the `Python package installer PIP <https://pypi.org/project/pip/>`_ , which internally handles also the compilation step.

Getting the source
------------------

You can directly download the code from the `source repository <https://gitlab.ethz.ch/nux/spring>`_ , or clone it on your machine with git:

.. code-block:: console

    $ git clone https://gitlab.ethz.ch/nux/spring.git
    
Installation
------------

.. warning:: 
    The compilation of source code *should* work also on Windows systems, but it is currently not tested.
    

Go with the terminal into the directory and run the pip install command as it follows:

.. code-block:: console

    $ python3 -m pip install .


The ``pip install`` command automatically takes care of both the compilation step (based on ``cmake``) and the installation of the spring module in the correct python folder. 

.. note:: 
    The *spring* module makes use of the `pybind11 <https://github.com/pybind/pybind11>`_ library to make the C++/CUDA compiled code and python interact. The library is automatically downloaded into the ``build`` directory when the ``pip install`` command is called. Plase, make sure to have a working internet connection at this point.

    

.. hint::

    In case of troubles with the installation process, please read the following section and/or run the ``pip`` command with the ``--verbose`` option, i.e.
    
    .. code-block::
    
        python3 -m pip install . --verbose
        
    

Checking for compilation errors
...............................

If the ``pip install`` command fails at compilation step, it could be quite hard to interpret the error message. In such a case, it is convenient to manually execute the compilation step by explicitly calling the ``cmake`` command. 
From the root of the source tree, create a ``build`` folder and run the ``cmake`` command as following:

.. code-block:: console
    
    $ mkdir build
    $ cd build
    $ cmake ..
    
At this step, cmake will look for packages/software needed for compilation. Missing tools will be indicated.

If successful, a proper ``Makefile`` is created. The actual compilation can be done achieved with the command:

.. code-block:: console
    
    $ make


Working on specific computing clusters
......................................

*Spring* is currently used for analyis on experimental data acquired at `FLASH <https://flash.desy.de/>`_, `EuXFEL <https://www.xfel.eu/>`_ and `SwissFEL <https://www.psi.ch/en/swissfel>`_.
Here are some instructions to make it run on the computing resources provided by those research facilities.

RA cluster (PSI)
^^^^^^^^^^^^^^^^

On `PSI Ra cluster <https://www.psi.ch/en/photon-science-data-services/offline-computing-facility-for-sls-and-swissfel-data-analysis>`_, first activate the desired conda environment


.. code-block:: console

    $ module load anaconda
    $ conda activate env_name
    

Load the following modules with the ``module`` command:

.. code-block:: console

    $ module load cuda/12.1.1
    
Navigate to the root of the spring repository and install with the usual ``pip`` command:

.. code-block:: console

    $ python3 -m pip install .

    
    
Maxwell cluster (FLASH/EuXFEL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the `Maxwell cluster <https://docs.desy.de/maxwell/>`_, the following configurations have been tested:

.. code-block:: console

    $ module load maxwell
    $ module load cuda/11.8
    
If you want to create a new environment:

.. code-block:: console

    $ module load conda/3.9
    $ conda create env_name
    
Navigate to the root of the spring repository and install with the usual ``pip`` command:

.. code-block:: console

    $ python3 -m pip install .
    
    
.. note::
    If you are using a new conda environment, don't forget to add it to Jupyter, such that it is listed among the available environments. Use the following commands with the environment activated:


    .. code-block:: console

        $ conda install -c anaconda ipykernel
        $ python3 -m ipykernel install --user --name=env_name

