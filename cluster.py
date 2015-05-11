# http://scikit-learn.org/stable/auto_examples/text/document_clustering.html#example-text-document-clustering-py
# http://brandonrose.org/clustering
#
# Author: Jimmy Zuber
# This script gathers information about the movies
# based on some criteria.
#
from __future__ import print_function
from collections import OrderedDict
from movie import Episode, Movie, MovieType
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from unidecode import unidecode
import argparse
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import re
import sys

# takes in a string containing a movie entry and returns a movie
# movies are formatted as such:
# where xxxxxx is the title of a movie:
# xxxxxx (year) year      = typical movie, year is year of movie (listed once in parens then once without)
# variations include:
# "xxxxxx"                = tv series
# "xxxxxx" (mini)         = tv mini-series
# (TV)                    = a tv movie, also on tv episodes
# (V)                     = a straight-to-video movie
# episodes have additional formatting:
# "xxxxxx" (year) {episode title (#season_number.episode_index_number)} year
# NOTE: since the current data doesn't have miniseries, they aren't supported
def parse_movie_line(line):
    # remove starting and trailing whitespace, check if empty line
    line = line.strip()
    if line == "":
        return None

    # don't consider suspended movies
    if "{{SUSPENDED}}" in line:
        return None

    # miniseries not supported
    if "(mini)" in line:
        print >> sys.stderr, 'Program does not support miniseries'
        return None

    # get the year of the movie
    year = line.split()[-1]
    year_end = None
    # for tv-series that span multiple years
    if "-" in year:
        year, year_end = year.split("-")
    if year != "????":
        year = int(year)

    # get the title, title is before (year) in line
    regex = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
    index = regex.search(line).end()
    title = line[:index]
    after = line[index:]

    # default movie type
    movie_type = MovieType.theater
    # if the movie is a tv-series
    if title[0] == "\"":
        movie_type = MovieType.tv_series
    elif "(TV)" in after:
        movie_type = MovieType.tv
    elif "(V)" in after:
        movie_type = MovieType.video
    elif "(VG)" in after:
        movie_type = MovieType.video_game

    return Movie(title, movie_type, year, year_end)

def read_people(filename, movies):
    # regex to find end of movie title
    regex = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
    people = {}
    with open(filename) as f:
        last_person = None
        for l in f:
            if len(l.strip()) == 0:
                last_person = None
                continue
            line = ""
            if not l.startswith('\t'):
                line = re.split('\t+', l)
                if len(line) >= 2:
                    last_person = line[0]
                    line = " ".join(line[1:])
                else:
                    raise ValueError("Line with bad format: {}".format(l))
            else:
                line = l.strip()

            try:
                index = regex.search(line).end()
                name = line[:index].strip()
            except:
                print(l)
                raise ValueError("Couldn't split line.")
            # only consider matching movies
            if not name in movies:
                continue
            if not last_person:
                raise ValueError("A movie was listed without an actor.")
            if not name in people:
                people[name] = []
            people[name].append(last_person)

    return people

def read_plots(filename, movies):
    regex = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
    plots = {}
    last_movie = None
    plot = ""

    with open(filename) as f:
        for l in f:
            # a new movie is encountered
            if l.startswith("MV: "):
                l = l[4:].strip()
                try:
                    index = regex.search(l).end()
                    name = l[:index].strip()
                except:
                    print(l)
                    raise ValueError("Couldn't split line.")

                # add the plot summary for the last movie
                if last_movie:
                    plots[last_movie] = plot

                # reset the plot
                plot = ""
                # consider this movie only if it is in the set of valid movies
                if name in movies:
                    last_movie = name
                else:
                    last_movie = None

            # if no valid movie is being considered, skip
            if not last_movie:
                continue

            # if more plot summary,
            if l.startswith("PL: "):
                l = l[4:]
                l = l.strip()
                plot += l + " "

        # add the plot summary for the last movie
        if last_movie:
            plots[last_movie] = plot

    return plots

def main():
    n_clusters = 20
    encoding = 'latin1'

    # takes a list of actions as command line arguments
    # source: http://stackoverflow.com/a/8527629
    class DefaultListAction(argparse.Action):
        CHOICES=['all', 'preprocess', 'features', 'cluster', 'svd', 'plot']
        def __call__(self, parser, namespace, values, option_string=None):
            if values:
                for value in values:
                    if value not in self.CHOICES:
                        message = ("invalid choice: {0!r} (choose from {1})"
                                .format(value, ', '.join([repr(action)
                                    for action in self.CHOICES])))

                        raise argparse.ArgumentError(self, message)
                setattr(namespace, self.dest, values)

    parser = argparse.ArgumentParser()
    parser.add_argument('actions', nargs='*', action=DefaultListAction,
                    default = ['plot'],
                    metavar='ACTION')
    args = parser.parse_args()

    if 'all' in args.actions or 'preprocess' in args.actions:
        # regex to find end of movie title
        regex = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
        # indices for movies and keywords
        movie_indices = {}
        movies = set()

        print("Reading movies...")
        with open("data/movies.list") as f:
            for line in f:
                movie = parse_movie_line(line)
                # invalid movies
                if not movie:
                    continue
                # movies of incorrect type
                if movie.movie_type != MovieType.theater:
                    print("Only considering regular movies.")
                    continue
                movies.add(movie.name)
                # set index of movie
                if not movie.name in movie_indices:
                    movie_indices[movie.name] = len(movie_indices)

        print('Reading plots...')
        plots = read_plots('data/plot.list', movies)

        print('Reading actors...')
        actors = read_people("data/actors.list", movies)

        print('Reading directors...')
        directors = read_people('data/directors.list', movies)

        print('Reading producers...')
        producers = read_people('data/producers.list', movies)

        print('Reading writers...')
        writers = read_people('data/writers.list', movies)

        print("Reading keywords...")
        keywords = {}
        keyword_counts = {}
        with open("data/keywords.list") as f:
            for line in f:
                index = regex.search(line).end()
                name = line[:index].strip()
                # only consider matching movies
                if not name in movies:
                    continue

                keyword = line.split()[-1]

                # add keyword to movie
                if not keyword in keyword_counts:
                    keyword_counts[keyword] = 0
                if not name in keywords:
                    keywords[name] = []
                keyword_counts[keyword] += 1
                keywords[name].append(keyword)

        print("Finding most frequent keywords...")
        most_frequent = [keyword for (keyword, _) in sorted(keyword_counts.items(), key=lambda (k, v): v, reverse=True)]

        n_removed = 25
        min_keywords = 5
        min_actors = 3
        print("""Eliminating {} most frequent keywords, removing movies with < {} keywords,
< {} actors, no directors, no producers, or no writers...""".format(n_removed, min_keywords, min_actors))
        print(most_frequent[:n_removed])
        for name, v in keywords.items():
            v = [k for k in v if not k in most_frequent[:n_removed]]
            test = name in actors and name in directors and name in producers
            test = test and name in writers and name in plots
            if not test or len(v) < min_keywords or len(actors[name]) < min_actors:
                movies.remove(name)
                del(keywords[name])
                if name in actors:
                    del(actors[name])
                if name in directors:
                    del(directors[name])
                if name in producers:
                    del(producers[name])
                if name in writers:
                    del(writers[name])
                if name in plots:
                    del(plots[name])

        print("Combining data sources...")
        data = {}
        for name in movies:
            data[name] = {
                'keywords': keywords[name],
                'actors': actors[name],
                'directors': directors[name],
                'producers': producers[name],
                'writers': writers[name],
                'plots': plots[name],
            }

        print("{} movies in dataset.".format(len(movies)))

        print('Writing data to disk...')
        joblib.dump(data, 'movie_data.pkl')

        if 'preprocess' in args.actions:
            args.actions.remove('preprocess')
            if len(args.actions) == 0:
                return
    else:
        print('Reading data from disk...')
        data = joblib.load('movie_data.pkl')

    if 'all' in args.actions or 'features' in args.actions:
        print("Computing tf-idf matrix...")
        # do tf-idf vectorization of actors and keywords
        actors_vectorizer    = TfidfVectorizer(analyzer=lambda d: d["actors"],    encoding=encoding, use_idf=False)
        directors_vectorizer = TfidfVectorizer(analyzer=lambda d: d["directors"], encoding=encoding, use_idf=False)
        keywords_vectorizer  = TfidfVectorizer(analyzer=lambda d: d["keywords"],  encoding=encoding)
        producers_vectorizer = TfidfVectorizer(analyzer=lambda d: d["producers"], encoding=encoding, use_idf=False)
        writers_vectorizer   = TfidfVectorizer(analyzer=lambda d: d["writers"],   encoding=encoding, use_idf=False)
        def plot_preprocessor(data):
            return unidecode(unicode(data['plots'], encoding)).lower()
        plots_vectorizer     = TfidfVectorizer(preprocessor=plot_preprocessor, encoding=encoding)
        union = FeatureUnion(
            [
                ('actors',    actors_vectorizer),
                ('directors', directors_vectorizer),
                ('keywords',  keywords_vectorizer),
                ('plots',     plots_vectorizer),
                ('producers', producers_vectorizer),
                ('writers',   writers_vectorizer),
            ],
            transformer_weights = {
                'actors':    1,
                'directors': 1,
                'keywords':  2,
                'plots':     2,
                'producers': 1,
                'writers':   1,
            }
        )

        X = union.fit_transform(data.values())

        terms = union.get_feature_names()
        print('Writing terms and transformed data to disk...')
        joblib.dump(terms, 'movie_terms.pkl')
        joblib.dump(X, 'movie_X.pkl')
        if 'features' in args.actions:
            args.actions.remove('features')
            if len(args.actions) == 0:
                return
    else:
        print('Reading terms and transformed data from disk...')
        terms = joblib.load('movie_terms.pkl')
        X = joblib.load('movie_X.pkl')

    if 'all' in args.actions or 'cluster' in args.actions:
        print("Clustering Data...")
        km = KMeans(n_clusters = n_clusters)
        km.fit(X)
        print('Writing clusters to disk...')
        joblib.dump(km, 'movie_cluster.pkl')

        if 'cluster' in args.actions:
            args.actions.remove('cluster')
            if len(args.actions) == 0:
                return
    else:
        print('Reading clusters from disk...')
        km = joblib.load('movie_cluster.pkl')

    if 'all' in args.actions or 'svd' in args.actions:
        print("Computing Lower Dimension Representation...")
        svd = TruncatedSVD(n_components=3).fit_transform(X)
        print('Writing svd to disk...')
        joblib.dump(svd, 'movie_svd.pkl')

        if 'svd' in args.actions:
            args.actions.remove('svd')
            if len(args.actions) == 0:
                return
    else:
        print('Reading svd from disk...')
        svd = joblib.load('movie_svd.pkl')


    if 'all' in args.actions or 'plot' in args.actions:
        n_clusters = len(km.cluster_centers_)
        N = len(km.labels_)

        xs, ys, zs = svd[:, 0], svd[:, 1], svd[:, 2]
        clusters = [[] for _ in range(n_clusters)]
        for i in range(N):
            clusters[km.labels_[i]].append((xs[i], ys[i], zs[i]))

        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        for i in range(n_clusters):
            print("Cluster {} ({} movies):".format(i, len(clusters[i])), end='')
            print(" ".join(terms[t] for t in order_centroids[i, :10]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        r = lambda: random.randint(0, 250)
        cluster_color = ['#%02X%02X%02X' % (r(),r(),r()) for _ in range(n_clusters)]

        ax.margins(0.05)
        for i, c in enumerate(clusters):
            xs, ys, zs = zip(*c)
            ax.scatter(xs, ys, zs, marker='o', color = cluster_color[i])

        plt.show()

main()
