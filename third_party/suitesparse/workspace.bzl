"""This module contains rules for the SuiteSparse library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

SUITESPARSE_VERSION = "7.8.3"
AMD_VERSION = "3.3.3"
BTF_VERSION = "2.3.2"
CAMD_VERSION = "3.3.3"
CCOLAMD_VERSION = "3.3.4"
COLAMD_VERSION = "3.3.4"
CHOLMOD_VERSION = "5.3.0"
KLU_VERSION = "2.3.5"
UMFPACK_VERSION = "6.3.5"

def suitesparse_workspace():
    http_archive(
        name = "suitesparse",
        build_file = "//third_party/suitesparse:suitesparse.BUILD",
        patches = ["//third_party/suitesparse:suitesparse.patch"],
        sha256 = "ce39b28d4038a09c14f21e02c664401be73c0cb96a9198418d6a98a7db73a259",
        strip_prefix = "SuiteSparse-{}".format(SUITESPARSE_VERSION),
        url = "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v{}.tar.gz".format(SUITESPARSE_VERSION),
    )
