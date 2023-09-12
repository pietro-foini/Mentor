# Mentor

## How to install

### Local virtual environment

We suggest to use [PyCharm Community](https://www.jetbrains.com/pycharm/download/#section=windows) for following 
steps 2-7.

1. Install Python 3.9: Make sure you have Python installed on your system. You can download it from the official Python 
website (https://www.python.org/) and follow the installation instructions for your operating system;
2. Clone the repository;
3. Create a virtual environment;
4. Activate the virtual environment;
5. Mark `src` folder as root directory;
6. Install project dependencies: 
   1. `pip install -r requirements.txt`
7. Look in (https://download.pytorch.org/whl/torch_stable.html and https://data.pyg.org/whl/torch-1.8.0%2Bcu111.html) for 
   the torch versions you want to install (torch version, python version, CUDA version, OS etc.). Add a URL dependency 
   to your `pyproject.toml` file. For example, the current .toml file has torch 1.8.0 working with GPU on Windows system;
8. Run the commands for further project dependencies:
   1. `poetry lock --no-update`
   2. `poetry install`

-----

## Dataset


## Model


## License

MIT

## Contacts

Please open an issue or contact pietro.foini1@gmail.com with any questions.