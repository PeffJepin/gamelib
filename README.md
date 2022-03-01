![Tests](https://github.com/PeffJepin/gamelib/actions/workflows/tests.yml/badge.svg)


# gamelib

This is a library I've been working on for writing 3d applications in python. You'll need to know at least a little bit about writing shaders in glsl to be effective with this library, though I do plan to include default implementations in the future.



### Prerequisites

This package requires support for OpenGL 3.3 or later and currently supports python >= 3.8.
If building from source some compiler will be required dependent on your platform.


### Installation

Optionally create a virtual environment:

Linux:

```sh
python3 -m venv venv
. venv/bin/activate
```

Windows:

```cmd
python3 -m venv venv
venv\Scripts\activate
```

Install with pip:

```sh
pip install gamelib
```

### Usage

Detailed usage documentation can be found in the modules themselves, to get started you should import gamelib and init must be called before doing anything involving the window or OpenGL context, so it's recommended to call at the entry point to your application. 

```py
import gamelib

gamelib.init()
```


Update should be called on a loop to not hang the application.

```py
while gamelib.is_running():
    gamelib.update()
```


To avoid hanging the main loop while executing a long running task, you can use the internal schedules.

```py
def my_long_running_function():
    time.sleep(1)

gamelib.threaded_schedule.once(my_long_running_function, -1)
```


As mentioned above, see the modules for further documentation, or refer to example applications linked below.


### Running the test suite

To run the tests you'll first need to get the source and install both the requirements and requirements-dev dependencies.

```sh
git clone https://github.com/peffjepin/gamelib.git
cd gamelib
python3 -m pip install -r requirements-dev.txt
python3 -m pip install -r requirements.txt
```

Then to run the tests simply:

```sh
pytest
```

For testing against different python versions you can run tox. The tox environment is set up to skip tests that require an opengl context.

```sh
tox
```

For a coverage report:

```sh
pytest --cov=gamelib --cov-report=html
```


### Example Applications

[gamelib-chess](https://github.com/peffjepin/gamelib-chess)
