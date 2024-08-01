"""This module contains rules for the SuiteSparse library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

SUITESPARSE_VERSION = "7.7.0"
AMD_VERSION = "3.3.2"
BTF_VERSION = "2.3.2"
CAMD_VERSION = "3.3.2"
CCOLAMD_VERSION = "3.3.3"
COLAMD_VERSION = "3.3.3"
CHOLMOD_VERSION = "5.2.1"
KLU_VERSION = "2.3.3"
UMFPACK_VERSION = "6.3.3"

def suitesparse_workspace():
    http_archive(
        name = "suitesparse",
        build_file = "//third_party/suitesparse:suitesparse.BUILD",
        patches = ["//third_party/suitesparse:suitesparse.patch"],
        sha256 = "529b067f5d80981f45ddf6766627b8fc5af619822f068f342aab776e683df4f3",
        strip_prefix = "SuiteSparse-{}".format(SUITESPARSE_VERSION),
        url = "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v{}.tar.gz".format(SUITESPARSE_VERSION),
    )
