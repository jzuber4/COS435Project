# gather.py
# Author: Jimmy Zuber
# This script gathers information about the movies
# based on some criteria.
#
from collections import OrderedDict
from movie import Episode, Movie, MovieType
from sklearn.feature_extraction.text import TfidfVectorizer
import math
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

def main():
    encoding = "latin1"
    # regex to find end of movie title
    regex = re.compile(r'\(\d\d\d\d(/([IVXLC]+))?\)|\(\?\?\?\?(/([IVXLC]+))?\)')
    # indices for movies and keywords
    movie_indices = {}
    # contian
    movies = OrderedDict()

    print("Reading in movies...")
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
            movies[movie.name] = []
            # set index of movie
            if not movie.name in movie_indices:
                movie_indices[movie.name] = len(movie_indices)

    print("Reading keywords...")
    with open("data/keywords.list") as f:
        for line in f:
            index = regex.search(line).end()
            name = line[:index].strip()
            # only consider matching movies
            if not name in movies:
                continue
            keyword = line[index:].strip()
            # add keyword to movie
            keywords = movies[name]
            keywords.append(keyword)

    print("Eliminating movies without keywords...")
    for name, v in movies.items():
        if not v:
            del movies[name]
        else:
            movies[name] = " ".join(v)

    # number of movies being clustered
    N = len(movies)
    print("Total movies: {}".format(N))

    print("Computing tf-idf matrix...")
    def tokenizer(string):
        return string.split()
    vectorizer = TfidfVectorizer(encoding=encoding, tokenizer=tokenizer)
    X = vectorizer.fit_transform(movies.items())

    print("Done!")







main()
