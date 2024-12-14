load("@//third_party/superlu:workspace.bzl", "SUPERLU_VERSION")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cmake(
    name = "superlu",
    cache_entries = select({
        "@platforms//os:osx": {
            "BUILD_SHARED_LIBS": "ON",
            "TPL_BLAS_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.dylib",
        },
        "@platforms//os:linux": {
            "BUILD_SHARED_LIBS": "ON",
            "TPL_BLAS_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.so",
        },
        "//conditions:default": {},
    }),
    lib_source = "@superlu//:all",
    out_shared_libs = select({
        "@platforms//os:osx": [
            "libsuperlu.dylib",
            "libsuperlu.{}.dylib".format(SUPERLU_VERSION.split(".", 1)[0]),
            "libsuperlu.{}.dylib".format(SUPERLU_VERSION),
        ],
        "@platforms//os:linux": [
            "libsuperlu.so",
            "libsuperlu.so.{}".format(SUPERLU_VERSION.split(".", 1)[0]),
            "libsuperlu.so.{}".format(SUPERLU_VERSION),
        ],
        "//conditions:default": [],
    }),
    deps = ["@openblas"],
)
