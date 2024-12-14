"""This module contains rules for the Protobuf library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

COM_GOOGLE_PROTOBUF_VERSION = "5.29.1"

def com_google_protobuf_workspace():
    http_archive(
        name = "com_google_protobuf",
        sha256 = "efeece317f91b93a0b4f4e9aabed0ac7580837f56283f1db81acdb07bd732778",
        strip_prefix = "protobuf-{}".format(COM_GOOGLE_PROTOBUF_VERSION),
        url = "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v{}.tar.gz".format(COM_GOOGLE_PROTOBUF_VERSION),
    )
