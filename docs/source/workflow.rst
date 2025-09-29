Typical workflow
-------------------

The *spring* module can be used to perform reconstructions in a standard *sequential* way, as described in the example :doc:`examples/Reconstruction`.

It is, however, possible (and suggested) to use asynchronous execution via the :meth:`spring.MPR.runasync` method to get full advantage from running in an interactive `Jupyter <https://jupyter.org/>`_ notebook, as presented in the example :doc:`examples/Async`.

In this section we enter more into the details of this mode, with some insights into the internal structure of the *spring* module.

Module import
^^^^^^^^^^^^^^


In the following examples, it is assumed that the following objects, i.e. :class:`spring.Pattern`, :class:`spring.Settings` and :class:`spring.MPR`,  are imported.


.. code-block:: python3

  from spring import Pattern, Settings, MPR

The control of asynchronous execution is achieved by importing the :data:`spring.hypervisor`.

.. code-block:: python3
  
  from spring import hypervisor as hv


It is important to note that :data:`spring.hypervisor` is **not a type/class**, but it is already an instance of :class:`spring._Hypervisor`.

Preparing data
^^^^^^^^^^^^^^

The step of data loading aims at creating a :class:`spring.Pattern` object that contains the experimental diffraction pattern. The usage is rather simple, as reported in :doc:`examples/Pattern`. 

In a typical workflow, it is convenient to create a function that loads the data from the desired file, which can be in different data formats, and returns a :class:`spring.Pattern` object ready for use. 

In the folder ``examples/data`` some exemplary diffraction patterns are prepared with a very simple structure. A function to load this type of data can be implemented as it follows:


.. code-block:: python3

  import numpy as np
  def load_data(filename):
      data = np.load(filename) # load raw data

      # create an instance of the Pattern object
      pattern = Pattern(pattern=data['pattern'], # diffraction data
                mask=data['mask'], # mask matrix, with values 0 or 1. Coordinates where mask==1 are excluded from the analysis
                center=data['center'], # coordinates of the center
                pid=data['pid'], # pulse id 
                cropsize=1024, # size to which the pattern is cropped
                size=256, # size to which the cropped pattern is rescaled
                satvalue=3e4) # pixel value above which pixels are considered to be saturated (especially relevant for pnCCD data)
    return pattern   

The implementation of a *data loader* function is specific of the data format. The ``data`` subfolder in the examples contains the following files:

.. code-block:: python3

  import os
  print(*sorted(os.listdir("data")))

.. code-block:: console

  example_A_1.npz example_A_2.npz example_A_3.npz example_A_4.npz example_B_1.npz example_B_2.npz example_B_3.npz example_B_4.npz

The example patterns can be now quickly loaded via the custom ``load_data`` function and the pattern displayed with:

.. code-block:: python3

  pattern = load_data("data/example_B_2.npz")
  pattern.plot()

The algorithm settings should be also defined. For more details see the :doc:`settings example <examples/Settings>` and the reference documentation of the :class:`spring.Settings` class.

.. code-block:: python3

  settings = Settings()

Live status of the process
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The methods ``hypervisor.livelog`` (see :meth:`spring._Hypervisor.livelog`) and  ``hypervisor.liveplot`` (see :meth:`spring._Hypervisor.liveplot`) allow one to get live information about the process. When run in a specific IPython cell, they update the output of that cell over time. For the ``liveplot`` usage, see :doc:`this example <examples/Async>`.

The ``livelog`` output before running any reconstruction looks like the following:

.. code-block:: python3

  hv.livelog()

  
.. code-block:: console

   




  - - - - - - - - - - -
  IDLE


When a new reconstruction process is launched with ``MPR(pattern=pattern, settings=settings).runasync()``, the output of the ``livelog`` is updated and the log of the MPR process is shown, along with a change of the status from ``IDLE`` to ``RUNNING`` accompained by the pulse-id (``pid``) of the pattern under analysis and a progressbar that informs about the proceeding of the reconstruction steps. 

.. code-block:: console


  Selected GPUs:
    - NVIDIA GeForce RTX 3090 (Id: 0, load: 0%)
  Initializing solver... Done.
  Initializing algorithms... Done.
  Initializing population... Done.
  ... Running main loop ...
  - - - - - - - - - - - - - - - - - - - - -
  RUNNING 19-7-14541946795 [###-------] 34%


.. hint::

  The ``livelog`` and ``liveplot`` functions updates their output from time to time in the cell they were called from. This may lead to some *jumps* in the visualization. Furthermore, the user typically wants to have their output always visible independently on the position in the ``.ipynb`` file. It is possible to create a dedicated view on a specific cell output on JupyterLab by right clicking with the mouse on the specific cell and selecting "Create New View for Cell Output". This will create a new page which only displays the specific cell output, which can be put on the side and always visible by tiling. See `this example <https://www.youtube.com/watch?v=Asa_ML45HP8>`_.
  It this then possible to hide the output of the live cell in the main document view by `clickng on the blue bar at the side of the cell output <https://www.youtube.com/watch?v=WgiX3ZRaTiY>`_.


Handling multiple reconstructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a typical workflow, the user wants to perform more than a single reconstruction. Several reconstructions can be achieved by multiple calls to the :meth:`spring.MPR.runasync` method.
For example, a reconstruction has already started by using data loaded from the ``examples/data/example_A_1.npy`` file:

.. code-block:: python3

  MPR(pattern=load_data("data/example_A_1.npz"), settings=sett).runasync()
  
The cell where the ``hv.livelog()`` was started will show something similar to:

.. code-block:: console

  Selected GPUs:
    - NVIDIA GeForce RTX 3090 (Id: 0, load: 0%)
  Initializing solver... Done.
  Initializing algorithms... Done.
  Initializing population... Done.
  ... Running main loop ...
  - - - - - - - - - - - - - - - - - - - - -
  RUNNING 197-10-14558540587 [#---------]  9%
  
When a new reconstruction is submitted, for example on ``example_A_2`` with the command

.. code-block:: python3

  MPR(pattern=load_data("data/example_A_2.npz"), settings=sett).runasync()
  
The ``hv.livelog()`` will still show the same output about the running reconstruction, but with the additional information that one reconstruction has been queued

.. code-block:: console

  Selected GPUs:
    - NVIDIA GeForce RTX 3090 (Id: 0, load: 0%)
  Initializing solver... Done.
  Initializing algorithms... Done.
  Initializing population... Done.
  ... Running main loop ...
  - - - - - - - - - - - - - - - - - - - - -
  RUNNING 197-10-14558540587 [###-------] 26% (1 queued)
  
The queued job will be executed as soon as the currently running one is completed. 
There is no restriction on the number of reconstructions that can be submitted, i.e. the length of the *execution queue* (apart from the total installed memory in the system).


For example, the execution of 

.. code-block:: python3

  for examplename in ["A_1", "A_2", "A_3", "A_4"]:
      MPR(pattern=load_data("data/example_"+ examplename+".npz"), settings=sett).runasync()

will make the ``livelog`` update to:      

.. code-block:: console

  Selected GPUs:
    - NVIDIA GeForce RTX 3090 (Id: 0, load: 0%)
  Initializing solver... Done.
  Initializing algorithms... Done.
  Initializing population... Done.
  ... Running main loop ...
  - - - - - - - - - - - - - - - - - - - - -
  RUNNING 197-10-14558540587 [#---------] 10% (3 queued)

while the execution of ``hv.info()`` will provide the following output:
  
.. code-block:: console

  Running: 197-10-14558540587
  Execution queue (3 total):
    0: 250-23-14565026066
    1: 71-1-14546865407
    2: 19-7-14541946795 

    
The call to :meth:`spring.MPR.runasync` (which is a wrapper to :meth:`spring._Hypervisor.append`) will put the reconstruction at the back of the execution queue. A reconstruction process can be put at the front of the execution queue with the :meth:`spring._Hypervisor.prepend` method, i.e.


.. code-block:: python3

  hv.prepend(MPR(pattern=load_data("data/example_B_1.npz"), settings=sett))

This will modify the output of ``hv.info()`` to:

.. code-block:: console

  Running: 197-10-14558540587
  Execution queue (4 total):
    0: 190-1769647347
    1: 250-23-14565026066
    2: 71-1-14546865407
    3: 19-7-14541946795


The execution of the current reconstruction can be interrupted via ``hv.stop()``. In this case, the next reconstruction in the queue is automatically started. 

Tagging reconstructions
"""""""""""""""""""""""
By default, reconstruction processes are identified by concatenatig their pid numbers (e.g. ``190-1769647347`` or ``250-23-14565026066``). However, it is sometimes necessary to perform multiple reconstruction procedures on the same diffraction data, for example to test the outcome as function of different values for the :class:`spring.Settings`. This can be achieved via the ``tag`` argument when creating the instance of :class:`spring.MPR`. This `tag`, if different from an empty string, is added to the reconstruction name (and the filename of the :class:`mpr.Result`).

In the following example, four different patterns are analyzed, tested using three different starting support sizes, for a total of 12 processes:

.. code-block:: python3

  for examplename in ["A_1", "A_2", "A_3", "A_4"]:
    for ss in [80,100,120]:
        sett = Settings().set('init','supportsize',ss)
        MPR(pattern=load_data("data/example_"+ examplename+".npz"), settings=sett, tag="suppsize:"+str(ss)).runasync()


The queue is, thus, the following:

.. code-block:: python3

  hv.info()

.. code-block:: console

  Running: 197-10-14558540587-suppsize:80
  Execution queue (11 total):
    0: 197-10-14558540587-suppsize:100
    1: 197-10-14558540587-suppsize:120
    2: 250-23-14565026066-suppsize:80
    3: 250-23-14565026066-suppsize:100
    4: 250-23-14565026066-suppsize:120
    5: 71-1-14546865407-suppsize:80
    6: 71-1-14546865407-suppsize:100
    7: 71-1-14546865407-suppsize:120
    8: 19-7-14541946795-suppsize:80
    9: 19-7-14541946795-suppsize:100
    10: 19-7-14541946795-suppsize:120


Removing jobs from the execution queue
""""""""""""""""""""""""""""""""""""""

The :data:`spring.hypervisor` object allows for further simple operations on the execution queue, and in particular it allows for the removal of submitted reconstructions via the method :meth:`spring._Hypervisor.remove`.

The ``hv.remove(item)`` method takes two different types as value for ``item``. In case of an **integer** ``i``, the element in position ``i`` of the queue given by ``hv.info()`` is removed. Please note that the item position in the queue may change while inputting the command.
In case of an ``item`` of type **string**, the full name of the job has to be given, e.g. ``item='71-1-14546865407-suppsize:120'``. This is more safe, as it is independent on the position in the queue.

.. note:: 

  The ``hv.remove()`` method only works on the process sitting in the execution queue. To cancel the running process, ``hv.stop()`` (see :meth:`spring._Hypervisor.stop()`) can be used. 
  
The **string** given as ``item`` parameter allows for the use of simple *wildcards* (the semantics follow the `fnmatch <https://docs.python.org/3/library/fnmatch.html>`_ module of the Python standard library). 
For example, it is possible to remove multiple items from the previous example (four patterns with three different configurations each) using the wildcards as it follows:

.. code-block:: python3

  # remove all jobs whose pid start with 71- 
  # (as the first number in the pid list is the run number
  # this is like saying "remove all reconstructions from Run 71")
  hv.remove("71-*") 

.. code-block:: console
  
  Removing 71-1-14546865407-suppsize:80 from queue
  Removing 71-1-14546865407-suppsize:100 from queue
  Removing 71-1-14546865407-suppsize:120 from queue

.. code-block:: python3

  # Remove all jobs whose name ends with "suppsize:100"
  hv.remove("*suppsize:100")
  
.. code-block:: console

  Removing 197-10-14558540587-suppsize:100 from queue
  Removing 250-23-14565026066-suppsize:100 from queue
  Removing 19-7-14541946795-suppsize:100 from queue

After these operations, the output of ``hv.info()`` looks like:

.. code-block:: console

  Running: 197-10-14558540587-suppsize:80
  Execution queue (5 total):
    0: 197-10-14558540587-suppsize:120
    1: 250-23-14565026066-suppsize:80
    2: 250-23-14565026066-suppsize:120
    3: 19-7-14541946795-suppsize:80
    4: 19-7-14541946795-suppsize:120
    
    
It is possible to kill all the queued jobs **and** the running one with:

.. code-block:: python3

  hv.kill()
  
  # Equivalent to:
  # hv.remove("*")
  # hv.stop()
    
  
.. code-block:: console

  Removing 197-10-14558540587 from queue
  Removing 250-23-14565026066 from queue
  Removing 71-1-14546865407 from queue
  Removing 19-7-14541946795 from queue
  Terminating running MPR process... Done


  
