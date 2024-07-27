"""This module contains rules for the Abseil C++ library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

ABSEIL_CPP_VERSION = "20240116.2"

def abseil_cpp_workspace():
    http_archive(
        name = "abseil-cpp",
        sha256 = "733726b8c3a6d39a4120d7e45ea8b41a434cdacde401cba500f14236c49b39dc",
        strip_prefix = "abseil-cpp-{}".format(ABSEIL_CPP_VERSION),
        url = "https://github.com/abseil/abseil-cpp/archive/refs/tags/{}.tar.gz".format(ABSEIL_CPP_VERSION),
    )
