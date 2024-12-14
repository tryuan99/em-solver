"""This module contains rules for the OpenBLAS library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

OPENBLAS_VERSION = "0.3.28"

def openblas_workspace():
    http_archive(
        name = "openblas",
        build_file = "//third_party/openblas:openblas.BUILD",
        sha256 = "f1003466ad074e9b0c8d421a204121100b0751c96fc6fcf3d1456bd12f8a00a1",
        strip_prefix = "OpenBLAS-{}".format(OPENBLAS_VERSION),
        url = "https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v{}.tar.gz".format(OPENBLAS_VERSION),
    )
