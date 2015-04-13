# gather.py
# Author: Jimmy Zuber
# This script gathers information about the movies
# based on some criteria.
#
from movie import Episode, Movie, MovieType
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
def parse_line(line):
    # remove starting and trailing whitespace, check if empty line
    line = line.strip()
    if line == "":
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
    try:
        year = int(year)
    except ValueError:
        raise ValueError('Year "{}" is not an integer'.format(year))

    # get the title, title is before (year) in line
    split = line.split('({})'.format(year))
    if len(split) != 2:
        raise ValueError('Line has ({}) in it {} times'.format(year, len(split) - 1))
    title = split[0].strip()
    after = split[1].strip()

    # default movie type
    movie_type = MovieType.theater
    # if the movie is a tv-series (episode
    if title[0] == "\"" and title[-1] == "\"":
        movie_type = MovieType.tv_series
        title = title[1:-1]
    elif "(TV)" in after:
        movie_type = MovieType.tv
    elif "(V)" in after:
        movie_type = MovieType.video

    # if its an episode
    if movie_type == MovieType.tv_series and "{" in after and "}" in after:
        _, after = after.split("{")
        after, _ = after.split("}")
        index = re.search(r'\(#\d+\.\d+\)', after).lastindex
        episode_title = after[:index]
        season_episode = after[index:]
        print(episode_title, season_episode)
        print(after)
        season, episode_index = season_episode[2:-1].split(".")
        try:
            season = int(season)
            episode_index = int(episode_index)
        except ValueError:
            raise ValueError('season "{}" and episode index "{}" not ints'.format(season, episode_index))
        return Episode(episode_title, title, season, episode_index, year)

    return Movie(title, movie_type, year, year_end)

def main():
    file_name = "data/movies.list" if len(sys.argv) < 2 else sys.argv[1]
    with open(file_name, encoding="latin1") as f:
        for l in f:
            parse_line(l)

main()
