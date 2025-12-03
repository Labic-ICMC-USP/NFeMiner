Installation
===============

To install the NFeMiner library, we recommend using **uv**, a fast and modern Python package manager that provides deterministic environments and manages dependencies, ensuring compatibility between NFeMiner and your existing environment.

A typical installation flow with uv looks like this:

.. code-block:: bash

    # Initialize a new project 
    # (if you already have a project, skip this step)
    uv init my-nfeminer-project
    cd my-nfeminer-project

    # Add NFeMiner as a dependency (GPU version)
    uv add "./NFeMiner[gpu]"

    # Alternatively, install the CPU-only version.
    # Use this if:
    # - you do not need GPU acceleration, OR
    # - the GPU dependencies are already installed, OR
    # - the GPU dependencies caused conflicts in your environment
    uv add "./NFeMiner"

    # Install and synchronize all dependencies
    uv sync


Alternative Installation Using pip
------------------------------------

You can also install NFeMiner using `pip`.

.. code-block:: bash

    # Install NFeMiner from a local directory (CPU version)
    pip install ./NFeMiner

    # Install NFeMiner from a local directory with GPU dependencies
    pip install "./NFeMiner[gpu]"
