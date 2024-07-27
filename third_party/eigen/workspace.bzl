"""This module contains rules for the Eigen library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

EIGEN_VERSION = "3.4.0"

def eigen_workspace():
    http_archive(
        name = "eigen",
        build_file = "//third_party/eigen:eigen.BUILD",
        sha256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72",
        strip_prefix = "eigen-{}".format(EIGEN_VERSION),
        url = "https://gitlab.com/libeigen/eigen/-/archive/{}/eigen-{}.tar.gz".format(EIGEN_VERSION, EIGEN_VERSION),
    )
