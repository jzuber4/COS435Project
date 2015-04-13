# movie.py
# Author: Jimmy Zuber
# This file contains the movie class
from enum import Enum

class MovieType(Enum):
  theater        = 0 # your ordinary movie, released in theaters
  tv_series      = 1 # a tv series
  tv_mini_series = 2 # a tv mini-series
  tv             = 3 # a movie released for tv
  video          = 4 # a movie released straight to video

class Movie():
  def __init__ (self, name, movie_type, year, year_end = None):
    self.name       = name
    self.movie_type = movie_type
    self.year       = year
    if self.movie_type == MovieType.tv_series \
            or self.movie_type == MovieType.tv_mini_series:
      self.year_end = year_end
      self.episodes = []


class Episode():
  def __init__ (self, name, movie_title, season, episode_index, year):
    self.name          = name
    self.movie_title   = movie_title
    self.season        = season
    self.episode_index = episode_index
    self.year          = year
