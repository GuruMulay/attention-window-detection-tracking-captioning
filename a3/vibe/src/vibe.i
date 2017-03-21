%module vibe

%include <opencv.i>
%cv_instantiate_all_defaults

%{
#include "vibe.hpp"
#include <random>
%}

%include "vibe.hpp"
