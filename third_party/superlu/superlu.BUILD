load("@//third_party/superlu:workspace.bzl", "SUPERLU_VERSION")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cmake(
    name = "superlu",
    generate_args = [
        "-DBUILD_SHARED_LIBS=ON",
        "-DTPL_BLAS_LIBRARIES=\"$EXT_BUILD_DEPS/openblas/lib/libopenblas.so\"",
    ],
    lib_source = "@superlu//:all",
    out_shared_libs = [
        "libsuperlu.so",
        "libsuperlu.so.{}".format(SUPERLU_VERSION.split(".", 1)[0]),
        "libsuperlu.so.{}".format(SUPERLU_VERSION),
    ],
    deps = ["@openblas"],
)
