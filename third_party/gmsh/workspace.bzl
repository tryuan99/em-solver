"""This module contains rules for the Gmsh library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

GMSH_VERSION = "4.13.1"

def gmsh_workspace():
    http_archive(
        name = "gmsh",
        build_file = "//third_party/gmsh:gmsh.BUILD",
        sha256 = "77972145f431726026d50596a6a44fb3c1c95c21255218d66955806b86edbe8d",
        strip_prefix = "gmsh-{}-source".format(GMSH_VERSION),
        url = "https://gmsh.info/src/gmsh-{}-source.tgz".format(GMSH_VERSION),
    )
