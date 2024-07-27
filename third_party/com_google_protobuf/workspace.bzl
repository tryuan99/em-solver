"""This module contains rules for the Protobuf library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

COM_GOOGLE_PROTOBUF_VERSION = "5.27.2"

def com_google_protobuf_workspace():
    http_archive(
        name = "com_google_protobuf",
        sha256 = "be42c6d2be30e0951797f2a94d73ba23806d8d177a82d32a26f7df38cc54cb14",
        strip_prefix = "protobuf-{}".format(COM_GOOGLE_PROTOBUF_VERSION),
        url = "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v{}.tar.gz".format(COM_GOOGLE_PROTOBUF_VERSION),
    )
