load(
    "@//third_party/suitesparse:workspace.bzl",
    "AMD_VERSION",
    "BTF_VERSION",
    "CAMD_VERSION",
    "CCOLAMD_VERSION",
    "CHOLMOD_VERSION",
    "COLAMD_VERSION",
    "KLU_VERSION",
    "SUITESPARSE_VERSION",
    "UMFPACK_VERSION",
)
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

KLU_SHARED_LIBS = [
    "libsuitesparseconfig.so",
    "libsuitesparseconfig.so.{}".format(SUITESPARSE_VERSION.split(".", 1)[0]),
    "libsuitesparseconfig.so.{}".format(SUITESPARSE_VERSION),
    "libamd.so",
    "libamd.so.{}".format(AMD_VERSION.split(".", 1)[0]),
    "libamd.so.{}".format(AMD_VERSION),
    "libbtf.so",
    "libbtf.so.{}".format(BTF_VERSION.split(".", 1)[0]),
    "libbtf.so.{}".format(BTF_VERSION),
    "libcamd.so",
    "libcamd.so.{}".format(CAMD_VERSION.split(".", 1)[0]),
    "libcamd.so.{}".format(CAMD_VERSION),
    "libccolamd.so",
    "libccolamd.so.{}".format(CCOLAMD_VERSION.split(".", 1)[0]),
    "libccolamd.so.{}".format(CCOLAMD_VERSION),
    "libcolamd.so",
    "libcolamd.so.{}".format(COLAMD_VERSION.split(".", 1)[0]),
    "libcolamd.so.{}".format(COLAMD_VERSION),
    "libcholmod.so",
    "libcholmod.so.{}".format(CHOLMOD_VERSION.split(".", 1)[0]),
    "libcholmod.so.{}".format(CHOLMOD_VERSION),
    "libklu.so",
    "libklu.so.{}".format(KLU_VERSION.split(".", 1)[0]),
    "libklu.so.{}".format(KLU_VERSION),
]

UMFPACK_SHARED_LIBS = [
    "libsuitesparseconfig.so",
    "libsuitesparseconfig.so.{}".format(SUITESPARSE_VERSION.split(".", 1)[0]),
    "libsuitesparseconfig.so.{}".format(SUITESPARSE_VERSION),
    "libamd.so",
    "libamd.so.{}".format(AMD_VERSION.split(".", 1)[0]),
    "libamd.so.{}".format(AMD_VERSION),
    "libcamd.so",
    "libcamd.so.{}".format(CAMD_VERSION.split(".", 1)[0]),
    "libcamd.so.{}".format(CAMD_VERSION),
    "libccolamd.so",
    "libccolamd.so.{}".format(CCOLAMD_VERSION.split(".", 1)[0]),
    "libccolamd.so.{}".format(CCOLAMD_VERSION),
    "libcolamd.so",
    "libcolamd.so.{}".format(COLAMD_VERSION.split(".", 1)[0]),
    "libcolamd.so.{}".format(COLAMD_VERSION),
    "libcholmod.so",
    "libcholmod.so.{}".format(CHOLMOD_VERSION.split(".", 1)[0]),
    "libcholmod.so.{}".format(CHOLMOD_VERSION),
    "libumfpack.so",
    "libumfpack.so.{}".format(UMFPACK_VERSION.split(".", 1)[0]),
    "libumfpack.so.{}".format(UMFPACK_VERSION),
]

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cmake(
    name = "klu",
    generate_args = [
        "-DSUITESPARSE_ENABLE_PROJECTS=\"klu\"",
        "-DBUILD_STATIC_LIBS=OFF",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBLAS_LIBRARIES=\"$EXT_BUILD_DEPS/openblas/lib/libopenblas.so\"",
        "-DLAPACK_LIBRARIES=\"$EXT_BUILD_DEPS/openblas/lib/libopenblas.so\"",
    ],
    lib_source = "@suitesparse//:all",
    out_shared_libs = KLU_SHARED_LIBS,
    deps = ["@openblas"],
)

cmake(
    name = "umfpack",
    generate_args = [
        "-DSUITESPARSE_ENABLE_PROJECTS=\"umfpack\"",
        "-DBUILD_STATIC_LIBS=OFF",
        "-DBUILD_SHARED_LIBS=ON",
        "-DBLAS_LIBRARIES=\"$EXT_BUILD_DEPS/openblas/lib/libopenblas.so\"",
        "-DLAPACK_LIBRARIES=\"$EXT_BUILD_DEPS/openblas/lib/libopenblas.so\"",
    ],
    lib_source = "@suitesparse//:all",
    out_shared_libs = UMFPACK_SHARED_LIBS,
    deps = ["@openblas"],
)
