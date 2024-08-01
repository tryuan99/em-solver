"""This module contains rules for the OpenBLAS library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

OPENBLAS_VERSION = "0.3.27"

def openblas_workspace():
    http_archive(
        name = "openblas",
        build_file = "//third_party/openblas:openblas.BUILD",
        sha256 = "aa2d68b1564fe2b13bc292672608e9cdeeeb6dc34995512e65c3b10f4599e897",
        strip_prefix = "OpenBLAS-{}".format(OPENBLAS_VERSION),
        url = "https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v{}.tar.gz".format(OPENBLAS_VERSION),
    )
