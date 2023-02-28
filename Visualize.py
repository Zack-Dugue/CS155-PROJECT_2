import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd

def visualize_factors(data,U,V):
    A, sigma, B = np.linalg.svd(np.transpose(V))

    V_2d = np.dot(np.transpose(A[:, :2]), np.transpose(V))
    U_2d = np.dot(np.transpose(A[:, :2]), np.transpose(U))

    movies = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/movies.csv")
    random_ten = [np.random.randint(0, len(movies) - 1) for i in range(10)]
    movie_list = np.array(movies)[:, 1]
    movie_list = movie_list[random_ten]
    coordinates = V_2d[:, random_ten]
    plt.figure()
    plt.grid()
    plt.suptitle("Random 10 Movies")
    plt.plot(coordinates[0, :], coordinates[1, :], "bo")
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")
    for i, txt in enumerate(movie_list):
        plt.text(coordinates[0, i], coordinates[1, i], txt)
    plt.show()

    # %%
    movies = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/movies.csv")
    pop_ten = data['Movie ID'].value_counts().index[:10]
    movie_list = np.array(movies)[:, 1]
    movie_list = movie_list[pop_ten]
    coordinates = V_2d[:, pop_ten]
    plt.figure()
    plt.grid()
    plt.suptitle("pop 10 movies")
    plt.plot(coordinates[0, :], coordinates[1, :], "bo")
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")
    for i, txt in enumerate(movie_list):
        plt.text(coordinates[0, i], coordinates[1, i], txt)
    plt.show()

    # %%
    movies = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/movies.csv")
    ten_best = data.groupby('Movie ID')['Rating'].mean().sort_values(ascending=False).index[:10]
    movie_list = np.array(movies)[:, 1]
    movie_list = movie_list[ten_best]
    coordinates = V_2d[:, ten_best]
    plt.figure()
    plt.suptitle("Ten Best Movies")
    plt.grid()
    plt.plot(coordinates[0, :], coordinates[1, :], "bo")
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")
    for i, txt in enumerate(movie_list):
        plt.text(coordinates[0, i], coordinates[1, i], txt)
    plt.show()
    # %%
    movies = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/movies.csv")
    movie_genre = 'Comedy'
    film_noir = movies[movies['Comedy'] == 1]['Movie ID'].values
    random_noir = np.random.choice(film_noir, 10)
    movie_list = np.array(movies)[:, 1]
    movie_list = movie_list[random_noir]
    coordinates = V_2d[:, random_noir]
    plt.figure()
    plt.grid()
    plt.suptitle(f"Random 10 from {movie_genre}")
    plt.plot(coordinates[0, :], coordinates[1, :], "bo")
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")
    for i, txt in enumerate(movie_list):
        plt.text(coordinates[0, i], coordinates[1, i], txt)
    plt.show()
    # %%
    movies = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/movies.csv")
    movie_genre = 'Sci-Fi'
    film_noir = movies[movies[movie_genre] == 1]['Movie ID'].values
    random_noir = np.random.choice(film_noir, 10)
    movie_list = np.array(movies)[:, 1]
    movie_list = movie_list[random_noir]
    coordinates = V_2d[:, random_noir]
    plt.figure()
    plt.grid()
    plt.suptitle(f"Random movies from {movie_genre}")
    plt.plot(coordinates[0, :], coordinates[1, :], "bo")
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")
    for i, txt in enumerate(movie_list):
        plt.text(coordinates[0, i], coordinates[1, i], txt)
    plt.show()
    # %%
    movies = pd.read_csv(
        "https://raw.githubusercontent.com/emiletimothy/Caltech-CS155-2023/main/miniproject2/data/movies.csv")
    movie_genre = 'Romance'
    film_noir = movies[movies[movie_genre] == 1]['Movie ID'].values
    random_noir = np.random.choice(film_noir, 10)
    movie_list = np.array(movies)[:, 1]
    movie_list = movie_list[random_noir]
    coordinates = V_2d[:, random_noir]
    plt.figure()
    plt.grid()
    plt.plot(coordinates[0, :], coordinates[1, :], "bo")
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")
    for i, txt in enumerate(movie_list):
        plt.text(coordinates[0, i], coordinates[1, i], txt)
    plt.suptitle(f"Random movies from {movie_genre}")
    plt.show()
