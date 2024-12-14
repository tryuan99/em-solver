"""This module contains rules for the SuperLU library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

SUPERLU_VERSION = "7.0.0"

def superlu_workspace():
    http_archive(
        name = "superlu",
        build_file = "//third_party/superlu:superlu.BUILD",
        sha256 = "d7b91d4e0bb52644ca74c1a4dd466a694ddf1244a7bbf93cb453e8ca1f6527eb",
        strip_prefix = "superlu-{}".format(SUPERLU_VERSION),
        url = "https://github.com/xiaoyeli/superlu/archive/refs/tags/v{}.tar.gz".format(SUPERLU_VERSION),
    )
