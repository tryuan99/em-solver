load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

OPENBLAS_VERSION = "0.3.29"

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cmake(
    name = "openblas",
    cache_entries = {
        "BUILD_SHARED_LIBS": "ON",
    },
    lib_source = "@openblas//:all",
    out_shared_libs = select({
        "@platforms//os:osx": [
            "libopenblas.dylib",
            "libopenblas.{}.dylib".format(OPENBLAS_VERSION.split(".", 1)[0]),
            "libopenblas.{}.dylib".format(OPENBLAS_VERSION.rsplit(".", 1)[0]),
        ],
        "@platforms//os:linux": [
            "libopenblas.so",
            "libopenblas.so.{}".format(OPENBLAS_VERSION.split(".", 1)[0]),
            "libopenblas.so.{}".format(OPENBLAS_VERSION.rsplit(".", 1)[0]),
        ],
        "//conditions:default": [],
    }),
)
