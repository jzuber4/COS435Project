# movie.py
# Author: Jimmy Zuber
# This file contains the movie class
class MovieType():
  theater        = 0 # your ordinary movie, released in theaters
  tv_series      = 1 # a tv series
  tv_mini_series = 2 # a tv mini-series
  tv             = 3 # a movie released for tv
  video          = 4 # a movie released straight to video
  video_game     = 5 # a video game movie

class Movie():
  def __repr__(self):
    s = "Movie- Name: {}, Year: {}, ".format(self.name, self.year)
    if hasattr(self, "year_end"):
        s += "Year End: {}, ".format(self.year_end)

    if self.movie_type == MovieType.theater:
        type_string = "Theater"
    elif self.movie_type == MovieType.tv_series:
        type_string = "TV Series"
    elif self.movie_type == MovieType.tv_mini_series:
        type_string = "TV Mini Series"
    elif self.movie_type == MovieType.tv:
        type_string = "TV"
    elif self.movie_type == MovieType.video:
        type_string = "Video"

    s += "Type: {}".format(type_string)
    return s

  def __init__ (self, name, movie_type, year, year_end = None):
    self.name       = name
    self.movie_type = movie_type
    self.year       = year
    if self.movie_type == MovieType.tv_series \
            or self.movie_type == MovieType.tv_mini_series:
      self.year_end = year_end
      self.episodes = []


class Episode():
  def __repr__(self):
      template = "Episode- Name: {}, Series Title: {}, Season: {}, Episode: {}, Year: {}"
      return template.format(self.name, self.series_title, self.season, self.episode_index, self.year)
  def __init__ (self, name, series_title, season, episode_index, year):
    self.name          = name
    self.series_title  = series_title
    self.season        = season
    self.episode_index = episode_index
    self.year          = year

