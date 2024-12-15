"""This module contains rules for the Protobuf library."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

COM_GOOGLE_PROTOBUF_VERSION = "5.28.3"

def com_google_protobuf_workspace():
    http_archive(
        name = "com_google_protobuf",
        sha256 = "7fce939b9b7181bd0bd157360e0cc88a8cabf01ac4efe4662494f56dd955d4c1",
        strip_prefix = "protobuf-{}".format(COM_GOOGLE_PROTOBUF_VERSION),
        url = "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v{}.tar.gz".format(COM_GOOGLE_PROTOBUF_VERSION),
    )
