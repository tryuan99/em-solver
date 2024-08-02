"""This module contains rules for the SuperLU library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

SUPERLU_VERSION = "6.0.1"

def superlu_workspace():
    http_archive(
        name = "superlu",
        build_file = "//third_party/superlu:superlu.BUILD",
        sha256 = "6c5a3a9a224cb2658e9da15a6034eed44e45f6963f5a771a6b4562f7afb8f549",
        strip_prefix = "superlu-{}".format(SUPERLU_VERSION),
        url = "https://github.com/xiaoyeli/superlu/archive/refs/tags/v{}.tar.gz".format(SUPERLU_VERSION),
    )
