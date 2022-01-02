#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = "Sergio Rossell"
SITENAME = "Abstractions and musings"
SITEDESCRIPTION = ""
# SITEURL = ""
SITEURL = "http://localhost:8000"

# plugins
PLUGIN_PATHS = ["plugins"]
PLUGINS = ["render_math", "i18n_subsites", "tipue_search"]
JINJA_ENVIRONMENT = {"extensions": ["jinja2.ext.i18n"]}

# theme and theme localization
THEME = "theme"
TIMEZONE = "Europe/Amsterdam"
DEFAULT_DATE_FORMAT = "%d %b %Y"
DEFAULT_LANG = "en"
LOCALE = "en_US"
STATIC_PATHS = ["images", "favicon.ico", "documents"]

# content paths
PATH = "content"
PAGE_PATHS = ["pages"]
ARTICLE_PATHS = ["blog"]

LOGO = "/images/logo.svg"
"./content/images/blog/tech/210126_splines2piecewise/pw_spline.png"
# special content
# NOTE. Interesting, but needs a good look into what images to show and maybe
# editing the display style.
# HERO = [
#     {
#         "image": "/images/blog/tech/190414_bda_and_odes/scatter.png",
#         "title": "Bayesian Data Analysis and ODEs",
#         "text": "Combining two fundamental paradigms to learn more from biological experiments",
#         "links": [],
#     },
#     {
#         "image": "/images/blog/tech/210126_splines2piecewise/pw_spline.png",
#         "title": "Splines as piecewise functions",
#         "text": "",
#         "links": [],
#     },
#     #    {
#     #        "image": "/images/hero/three.png",
#     #        "title": "Third plot",
#     #        "text": "",
#     #        "links": [],
#     #    },
# ]

# Social widget
SOCIAL = (
    ("linkedin", "https://www.linkedin.com/in/sergiorossell/"),
    ("github", "https://github.com/srossell"),
)

ABOUT = {
    "image": "/images/about/about.jpeg",
    "mail": "sergio.rossell gmail com",
    # keep it a string if you dont need multiple languages
    "text": "drop me a line.",
}

# navigation and homepage options
DISPLAY_PAGES_ON_MENU = True
DISPLAY_PAGES_ON_HOME = False
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_TAGS_ON_MENU = False
USE_FOLDER_AS_CATEGORY = True
PAGE_ORDER_BY = "order"

MENUITEMS = [
    ("Blog", "blog.html"),
    ("Categories", "categories.html"),
    ("Tags", "tags.html"),
]

DIRECT_TEMPLATES = [
    "index",
    "blog",
    "categories",
    "tags",
    "search",  # needed for tipue_search plugin
]

# setup disqus
DISQUS_SHORTNAME = "gitcd-dev"
DISQUS_ON_PAGES = False  # if true its just displayed on every static page, like this you can still enable it per page

# setup google maps
# GOOGLE_MAPS_KEY = "AIzaSyDtLsg5TViozEWlHg4RihNiuUv9T8IPm90"
# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True
