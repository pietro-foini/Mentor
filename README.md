# Mentor

This is a PyTorch implementation of Mentor (📖 [Modeling Teams Performance Using Deep Representational Learning on Graphs](https://arxiv.org/abs/2206.14741))

Authors: Pietro Foini, Francesco Carli, Nicolò Gozzi, Nicola Perra, Rossano Schifanella

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

Now you're all set! 🎉 Happy coding! 😄✨

-----

## Dataset

The datasets used are divided into two main categories: **synthetic** and **real-world** datasets. Synthetic data has been 
generated in such a way as to systematically validate the theoretical assumptions regarding the key contributions of 
the three effects: *topology*, *centrality*, and *position*. Real-world datasets, on the other hand, have been employed to 
assess the effectiveness of the models thus developed.

For more details on both types of datasets, we refer you to the respective folder where the analyses related to them have been included.

### Synthetic

### Real-world

Freely accessible and copyright-free data concerning team management and their respective performance are lacking. 
The three datasets we have focused on are as follows:

<table align="center">
  <tr>
    <td style="text-align: center; padding: 20px;">
      <img src="./src/datasets/real-world/Dribbble/logo.png" width="200" /><br>
      <a href="https://example.com/dribbble" style="text-align: center; display: block;">Dribbble</a>
    </td>
    <td style="text-align: center; padding: 20px;">
      <img src="./src/datasets/real-world/Kaggle/logo.png" width="200" /><br>
      <a href="https://example.com/kaggle" style="text-align: center; display: block;">Kaggle</a>
    </td>
    <td style="text-align: center; padding: 20px;">
      <img src="./src/datasets/real-world/IMDb/logo.png" width="200" /><br>
      <a href="https://example.com/imdb" style="text-align: center; display: block;">IMDb</a>
    </td>
  </tr>
</table>

## Model


## License

MIT

## Contacts

Please open an issue or contact pietro.foini1@gmail.com with any questions.