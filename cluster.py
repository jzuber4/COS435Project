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

def read_people(people, filename, movies):
    # regex to find end of movie title
    regex = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
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
            # only add new names
            if not last_person in people[name]:
                people[name].append(last_person)

    return people

# prepicked sets of movies that should be clustered together
def get_match_sets():
    return [
        ["Spider-Man (2002)", "X-Men (2000)", "X-Men Origins: Wolverine (2009)"], # early superhero films
        ["The Lion King (1994)", "Toy Story (1995)", "Monsters, Inc. (2001)"], # 3d animated disney / pixar
        ["City Lights (1931)", "The Gold Rush (1925)", "Modern Times (1936)"], # Chaplin
        ["Saw (2004)", "The Ring (2002)", "Paranormal Activity (2007)"], # horror movies
        ["The Notebook (2004)", "Dear John (2010/I)", "A Walk to Remember (2002)"], # romantic dramas
        ["The Terminator (1984)", "Aliens (1986)", "Predator (1987)"], # 80s sci fi
        ["Zoolander (2001)", "Blades of Glory (2007)", "Tropic Thunder (2008)"], # comedy
        ["Notting Hill (1999)", "Love Actually (2003)", "Music and Lyrics (2007)"], # romantic comedies
        ["Moonrise Kingdom (2012)", "The Darjeeling Limited (2007)", "Rushmore (1998)"], # Wes Anderson
        ["Jack Reacher (2012)", "Mission: Impossible (1996)", "Minority Report (2002)"], # Tom Cruise Action Movies
    ]

def read_genres(filename, movies):
    regex = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
    genres = {}
    with open(filename) as f:
        for l in f:
            l = l.strip()
            genre = l.split('\t')[-1]
            movie = l[:regex.search(l).end()]
            if not movie in movies:
                continue
            if not movie in genres:
                genres[movie] = []
            genres[movie].append(genre)
    return genres


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
    encoding = 'latin1'

    # takes a list of actions as command line arguments
    # source: http://stackoverflow.com/a/8527629
    class DefaultListAction(argparse.Action):
        CHOICES=['all', 'preprocess', 'features', 'cluster', 'svd', 'plot', 'test']
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
                    metavar='ACTION',
                    help='Optionally, actions from the list: {}.'.format(DefaultListAction.CHOICES))
    parser.add_argument('-q', '--quiet', action='store_true', help='disable printing of progress')
    parser.add_argument('-l', '--limit', type=float, help='fraction of movies to consider', default = 1.0)
    parser.add_argument('-c', '--clusters', type=int, help='number of clusters', default = 20)
    args = parser.parse_args()
    if not 0 < args.limit  <= 1.0:
        parser.print_help()
        return

    n_clusters = args.clusters

    # make printing function that prints if not args.quiet
    def optional_print(if_print):
        def inner(*args, **kwargs):
            if if_print:
                print(*args, **kwargs)
        return inner
    q_print = optional_print(not args.quiet)

    if 'all' in args.actions or 'preprocess' in args.actions:
        # regex to find end of movie title
        regex = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
        movies = set()

        q_print("Reading movies...")
        with open("data/movies.list") as f:
            for line in f:
                movie = parse_movie_line(line)
                # invalid movies
                if not movie:
                    continue
                # movies of incorrect type
                if movie.movie_type != MovieType.theater:
                    q_print("Only considering regular movies.")
                    continue
                movies.add(movie.name)

        q_print('Reading genres...')
        genres = read_genres('data/genres.list', movies)

        q_print('Reading plots...')
        plots = read_plots('data/plot.list', movies)

        q_print('Removing accents, lowercasing words in plots...')
        for name, plot in plots.items():
            plots[name] = unidecode(unicode(plot, encoding)).lower()

        q_print('Reading people: ', end='')
        sys.stdout.flush()

        q_print('actors... ', end='')
        sys.stdout.flush()
        people = read_people({}, "data/actors.list", movies)

        q_print('editors... ', end='')
        sys.stdout.flush()
        people = read_people(people, "data/editors.list", movies)

        q_print('directors... ', end='')
        sys.stdout.flush()
        people = read_people(people, 'data/directors.list', movies)

        q_print('producers... ', end='')
        sys.stdout.flush()
        people = read_people(people, 'data/producers.list', movies)

        q_print('writers... ')
        people = read_people(people, 'data/writers.list', movies)

        q_print("Reading keywords...")
        keywords = {}
        with open("data/keywords.list") as f:
            for line in f:
                index = regex.search(line).end()
                name = line[:index].strip()
                # only consider matching movies
                if not name in movies:
                    continue

                keyword = line.split()[-1]

                # add keyword to movie
                if not name in keywords:
                    keywords[name] = []
                keywords[name].append(keyword)

        bad_substrings = ['title', 'novel', 'book', 'play', 'film', 'relationship']
        q_print("Eliminating keywords with substrings in {}".format(bad_substrings))
        for name, v in keywords.items():
            keywords[name] = [keyword for keyword in v if not any(s in keyword for s in bad_substrings)]

        q_print("Combining data sources, only including movies with people, plots, genres and keywords...")
        data = OrderedDict()
        for name in movies:
            if name in people and name in plots and name in keywords and name in genres:
                data[name] = {
                    'genres':    genres[name],
                    'keywords':  keywords[name],
                    'people':    people[name],
                    'plots':     plots[name],
                }

        q_print("{} movies in dataset.".format(len(data)))

        q_print('Writing data to disk...')
        joblib.dump(data, 'movie_data.pkl')

        if 'preprocess' in args.actions:
            args.actions.remove('preprocess')
            if len(args.actions) == 0:
                return
    else:
        q_print('Reading data from disk...')
        data = joblib.load('movie_data.pkl')

    if 'all' in args.actions or 'features' in args.actions:
        if args.limit < 1.0:
            # exempt certain movies from removal
            exempted_movies = set()
            for movie_list in get_match_sets():
                for m in movie_list:
                    exempted_movies.add(m)

            q_print("Restricting dataset to ~{}% of original".format(100 * args.limit))
            for name in data.keys():
                if random.random() > args.limit and not name in exempted_movies:
                    del(data[name])

            q_print("{} movies in dataset.".format(len(data)))

        q_print('Writing the data that will be used')
        joblib.dump(data, 'movie_data_in_use.pkl')

        q_print("Computing tf-idf matrix...")
        # do tf-idf vectorization of actors and keywords
        keywords_vectorizer = TfidfVectorizer(analyzer=lambda d: d["keywords"], encoding=encoding)
        people_vectorizer = TfidfVectorizer(analyzer=lambda d: d["people"], encoding=encoding, use_idf= False)
        genres_vectorizer = TfidfVectorizer(analyzer=lambda d: d["genres"], encoding=encoding, use_idf= False)
        plots_vectorizer = TfidfVectorizer(preprocessor=lambda d: d["plots"], encoding=encoding, stop_words='english')
        union = FeatureUnion(
            [
                ('genres',    genres_vectorizer),
                ('keywords',  keywords_vectorizer),
                ('people',    people_vectorizer),
                ('plots',     plots_vectorizer),
            ],
            transformer_weights = {
                'genres':    2,
                'keywords':  4,
                'people':    1,
                'plots':     1,
            }
        )

        X = union.fit_transform(data.values())

        terms = union.get_feature_names()
        q_print('Writing terms and transformed data to disk...')
        joblib.dump(terms, 'movie_terms.pkl')
        joblib.dump(X, 'movie_X.pkl')
        if 'features' in args.actions:
            args.actions.remove('features')
            if len(args.actions) == 0:
                return
    else:
        q_print('Reading terms and transformed data from disk...')
        terms = joblib.load('movie_terms.pkl')
        X = joblib.load('movie_X.pkl')

    if 'all' in args.actions or 'cluster' in args.actions:
        q_print("Clustering Data...")
        km = KMeans(n_clusters = n_clusters)
        km.fit(X)
        q_print('Writing clusters to disk...')
        joblib.dump(km, 'movie_cluster.pkl')

        if 'cluster' in args.actions:
            args.actions.remove('cluster')
            if len(args.actions) == 0:
                return
    else:
        q_print('Reading clusters from disk...')
        km = joblib.load('movie_cluster.pkl')

    if 'all' in args.actions or 'svd' in args.actions:
        q_print("Computing Lower Dimension Representation...")
        svd = TruncatedSVD(n_components=3).fit_transform(X)
        q_print('Writing svd to disk...')
        joblib.dump(svd, 'movie_svd.pkl')

        if 'svd' in args.actions:
            args.actions.remove('svd')
            if len(args.actions) == 0:
                return
    else:
        q_print('Reading svd from disk...')
        svd = joblib.load('movie_svd.pkl')


    if 'all' in args.actions or 'plot' in args.actions:
        n_clusters = len(km.cluster_centers_)
        N = len(km.labels_)

        xs, ys, zs = svd[:, 0], svd[:, 1], svd[:, 2]
        clusters = [[] for _ in range(n_clusters)]
        for i in range(N):
            clusters[km.labels_[i]].append((xs[i], ys[i], zs[i]))

        q_print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        for i in range(n_clusters):
            q_print("Cluster {} ({} movies):".format(i, len(clusters[i])), end='')
            q_print(" ".join(terms[t] for t in order_centroids[i, :10]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        r = lambda: random.randint(0, 252)
        cluster_color = ['#%02X%02X%02X' % (r(),r(),r()) for _ in range(n_clusters)]

        ax.margins(0.05)
        for i, c in enumerate(clusters):
            xs, ys, zs = zip(*c)
            ax.scatter(xs, ys, zs, marker='o', color = cluster_color[i])

        plt.show()


    if 'all' in args.actions or 'test' in args.actions:
        q_print("Reading in data that was used for computation")
        data = joblib.load('movie_data_in_use.pkl')

        q_print("Reading in movie links")
        seen = set()
        groups = []
        with open("data/movie-links.list") as f:
            group = []
            regex_begin = re.compile('\((follows |followed by |version of |alternate language version of )')
            regex_end = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
            for l in f:
                if len(l.strip()) == 0:
                    if len(group) != 0:
                        groups.append(group)
                        group = []
                    continue


                if l.startswith(" "):
                    l = l[regex_begin.search(l).end():]

                l = l.strip()

                movie = l[:regex_end.search(l).end()]
                if not movie in seen and movie in data:
                    seen.add(movie)
                    group.append(movie)

        groups = filter(lambda g: len(g) > 1, groups)

        indices = {}
        for i, name in enumerate(data.keys()):
            indices[name] = i

        q_print("Comparing movie links to clusters.")
        count = 0
        for group in groups:
            first = km.labels_[indices[group[0]]]
            if all(km.labels_[indices[movie]] == first for movie in group[1:]):
                count += 1

        q_print("{} out of {} movie links are contained by clusters.".format(count, len(groups)))

        # hand picked sets of movies that ought to be clustered together
        q_print("Comparing pre-picked movies to clusters.")
        count = 0
        for group in get_match_sets():
            first = km.labels_[indices[group[0]]]
            if all(km.labels_[indices[movie]] == first for movie in group[1:]):
                count += 1

        q_print("{} out of {} pre-picked groups correspond to clusters.".format(count, len(get_match_sets())))


main()
