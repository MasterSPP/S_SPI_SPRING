Hardware management
===================


Overview
--------
Due to the heavy computational load of the method, the software requires the presence of at least one NVIDIA GPU.
The *spring* module is built upon three software layers:

1. The upper layer is written in Python, and it is directly importable by the user via the objects :py:meth:`spring.Pattern`, :py:meth:`spring.Settings`, :py:meth:`spring.Result` and :py:meth:`spring.MPR`.
2. Internally, the :py:meth:`spring.MPR` object calls functions written in C++, accessible by Python thanks to the `PyBind11 <https://github.com/pybind/pybind11>`_ library. 
3. Computationally heavy tasks, most of which are related to the execution of *iterative phase retrieval algorithms*, are offloaded to GPUs with source code written in `CUDA <https://developer.nvidia.com/cuda-toolkit>`_ language.


Hardware settings
-----------------
The *spring* module takes advantage of both **multiple computing cores** and **multiple GPUs** as long as they belong to the same *computing node* (i.e., they have a shared memory space) by parallelizing computations. Parallelization on *distributed memory systems* is currently not supported. 

There are two parameters in :py:meth:`spring.Settings` that can be tuned to optimize the performance, i.e. the time to solution. Those are ``threads`` and ``gpus`` in the ``global`` section of the :py:meth:`spring.Settings` object (see also :doc:`examples/Settings`). 

``threads``:
    This value indicates the number of CPU threads to use **per GPU**. For example, if ``threads=8`` and two GPUs are selected for computation, the total number of CPU threads actually used is 16. Having more than a single thread per GPU is convenient for two reasons:
    
    1. While most of the work is offloaded to GPUs, there are still some calculations performed on the CPU side that can be boosted by parallel CPU execution.
    2. The offloading of calculations to GPUs is controlled by the CPU, that has to preare the data and transfer it back and forth from the system memory to the GPU memory. Data transfer can then be optimized by having multiple threads handling calculations on a single GPU, as communication and calculations can be overlapped. 
    
    Depending on the GPU and CPU models, the optimal number of threads may vary. A number of threads equal to 8 is tipically a good compromise for high-end GPUs.
    
    .. note::
        The user should ensure that the total number of used threads is not exceeding the number of physical computing cores available. If a higher number of threads is set, this tipycally brings a significat drop in performance due to overheads.
    
    
``gpus``:
    This parameter is important in the case of a system equipped with multiple GPUS. It can be either an integer number or a list of positive integers. The behavior is the following:
    
    **value**:
        Example: ``gpus=2``. In this case, two GPUs are used, and are selected by their GPU id (i.e. GPU-0 and GPU-1). If ``gpus<0`` or ``gpus`` exceeds the number of available GPUs, all GPUs detected are selected for computation. 
        
    .. note::
        This behavior can be changed by activating a special flag at compilation time, to select the ``N`` gpus with lowest current computing load instead of the *first* ``N`` gpus available. Further details are given in `Unmanaged multi-GPU systems`_.
        
    **list**:
        Example: ``gpus=[0,2,3]``. In this case, the specific GPU ids are selected, i.e. GPU-0, GPU-2 and GPU-3 are selected. 
        
    .. note::
        The settings ``gpus=2`` and ``gpus=[2]`` have different outcomes. In the first case, the first two GPUs are selected (GPU-0 and GPU-1). In the second, only GPU-2 (typically the third one after GPU-0 and GPU-1) is selected.
        
    For computing environments managed by job schedulers (like `SLURM <https://slurm.schedmd.com/documentation.html>`_), it is often convenient to set ``gpus=-1``, because even on a multi-GPU computing node only the requested GPUs are visible.
    

Unmanaged multi-GPU systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For *unmanaged* computing nodes, i.e. those where the distribution of resources is not demanded to a job scheduler (e.g. `SLURM <https://slurm.schedmd.com/documentation.html>`_), the users have to agree on the use of computing resources. This may be tricky especially in the case of a multi-GPU system with multiple users. 

The outcome of the ``gpus=N`` parameter in :py:meth:`spring.Settings` can be changed to select the *least loaded* N GPUs instead of the ones with the first N GPU ids. This allows for a more flexible and optimal sharing of computing resources on a single multi-GPU computing node. 

This option has to be enabled at installation time (see also :doc:`install`) by setting the environment variable ``WITH_NVCC=On``. The installation via ``pip`` is then

.. code:: console

    $ WITH_NVML=On python3 -m pip install .
    
    
In this way, when the reconstruction is launched via the :meth:`spring.MPR.run` or :meth:`spring.MPR.runasync`, the current load on the available GPUs is inspected and those with less computing load are selected.

.. note::
    The option ``WITH_NVML=On`` requires the presence of the `NVIDIA Management Library <https://developer.nvidia.com/management-library-nvml>`_ on the system. This means that the header files must be reachable at compilation time and the corresponding shared library must be reachable at runtime. 


Systems without GPUs
^^^^^^^^^^^^^^^^^^^^

At the current status, the core of *spring* is **only implemented as GPU code**, so it is not possible to run the imaging algorithm on systems without GPUs. 
For testing purposes, there are however two options that, while not allowing actual reconstructions on CPU-only systems, enable the user to test the installation process and use all *spring* objects apart from :class:`spring.MPR`.

    1. It is possible to install the CUDA compiler even on systems not equipped with GPUs. 
    
    2. It is possible to compile the code without a proper CUDA compiler, by setting the enironment variable ``WITH_CUDA=Off`` before installation with pip, i.e.
    
    .. code-block:: console
    
        WITH_CUDA=Off python3 -m pip install .
    
        
The user will get an error at runtime if a reconstruction is attemped.


