load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

AMD_VERSION = "3.3.3"

BTF_VERSION = "2.3.2"

CAMD_VERSION = "3.3.3"

CCOLAMD_VERSION = "3.3.4"

COLAMD_VERSION = "3.3.4"

CHOLMOD_VERSION = "5.3.2"

KLU_VERSION = "2.3.5"

UMFPACK_VERSION = "6.3.5"

KLU_SHARED_LIBS = select({
    "@platforms//os:osx": [
        "libsuitesparseconfig.dylib",
        "libsuitesparseconfig.{}.dylib".format(SUITESPARSE_VERSION.split(".", 1)[0]),
        "libsuitesparseconfig.{}.dylib".format(SUITESPARSE_VERSION),
        "libamd.dylib",
        "libamd.{}.dylib".format(AMD_VERSION.split(".", 1)[0]),
        "libamd.{}.dylib".format(AMD_VERSION),
        "libbtf.dylib",
        "libbtf.{}.dylib".format(BTF_VERSION.split(".", 1)[0]),
        "libbtf.{}.dylib".format(BTF_VERSION),
        "libcamd.dylib",
        "libcamd.{}.dylib".format(CAMD_VERSION.split(".", 1)[0]),
        "libcamd.{}.dylib".format(CAMD_VERSION),
        "libccolamd.dylib",
        "libccolamd.{}.dylib".format(CCOLAMD_VERSION.split(".", 1)[0]),
        "libccolamd.{}.dylib".format(CCOLAMD_VERSION),
        "libcolamd.dylib",
        "libcolamd.{}.dylib".format(COLAMD_VERSION.split(".", 1)[0]),
        "libcolamd.{}.dylib".format(COLAMD_VERSION),
        "libcholmod.dylib",
        "libcholmod.{}.dylib".format(CHOLMOD_VERSION.split(".", 1)[0]),
        "libcholmod.{}.dylib".format(CHOLMOD_VERSION),
        "libklu.dylib",
        "libklu.{}.dylib".format(KLU_VERSION.split(".", 1)[0]),
        "libklu.{}.dylib".format(KLU_VERSION),
    ],
    "@platforms//os:linux": [
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
    ],
    "//conditions:default": [],
})

UMFPACK_SHARED_LIBS = select({
    "@platforms//os:osx": [
        "libsuitesparseconfig.dylib",
        "libsuitesparseconfig.{}.dylib".format(SUITESPARSE_VERSION.split(".", 1)[0]),
        "libsuitesparseconfig.{}.dylib".format(SUITESPARSE_VERSION),
        "libamd.dylib",
        "libamd.{}.dylib".format(AMD_VERSION.split(".", 1)[0]),
        "libamd.{}.dylib".format(AMD_VERSION),
        "libcamd.dylib",
        "libcamd.{}.dylib".format(CAMD_VERSION.split(".", 1)[0]),
        "libcamd.{}.dylib".format(CAMD_VERSION),
        "libccolamd.dylib",
        "libccolamd.{}.dylib".format(CCOLAMD_VERSION.split(".", 1)[0]),
        "libccolamd.{}.dylib".format(CCOLAMD_VERSION),
        "libcolamd.dylib",
        "libcolamd.{}.dylib".format(COLAMD_VERSION.split(".", 1)[0]),
        "libcolamd.{}.dylib".format(COLAMD_VERSION),
        "libcholmod.dylib",
        "libcholmod.{}.dylib".format(CHOLMOD_VERSION.split(".", 1)[0]),
        "libcholmod.{}.dylib".format(CHOLMOD_VERSION),
        "libumfpack.dylib",
        "libumfpack.{}.dylib".format(UMFPACK_VERSION.split(".", 1)[0]),
        "libumfpack.{}.dylib".format(UMFPACK_VERSION),
    ],
    "@platforms//os:linux": [
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
    ],
    "//conditions:default": [],
})

filegroup(
    name = "all",
    srcs = glob(["**"]),
)

cmake(
    name = "klu",
    cache_entries = select({
        "@platforms//os:osx": {
            "SUITESPARSE_ENABLE_PROJECTS": "klu",
            "BUILD_STATIC_LIBS": "OFF",
            "BUILD_SHARED_LIBS": "ON",
            "BLAS_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.dylib",
            "LAPACK_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.dylib",
        },
        "@platforms//os:linux": {
            "SUITESPARSE_ENABLE_PROJECTS": "klu",
            "BUILD_STATIC_LIBS": "OFF",
            "BUILD_SHARED_LIBS": "ON",
            "BLAS_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.so",
            "LAPACK_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.so",
        },
        "//conditions:default": {},
    }),
    lib_source = "@suitesparse//:all",
    out_shared_libs = KLU_SHARED_LIBS,
    deps = ["@openblas"],
)

cmake(
    name = "umfpack",
    cache_entries = select({
        "@platforms//os:osx": {
            "SUITESPARSE_ENABLE_PROJECTS": "umfpack",
            "BUILD_STATIC_LIBS": "OFF",
            "BUILD_SHARED_LIBS": "ON",
            "BLAS_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.dylib",
            "LAPACK_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.dylib",
        },
        "@platforms//os:linux": {
            "SUITESPARSE_ENABLE_PROJECTS": "umfpack",
            "BUILD_STATIC_LIBS": "OFF",
            "BUILD_SHARED_LIBS": "ON",
            "BLAS_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.so",
            "LAPACK_LIBRARIES": "$$EXT_BUILD_DEPS/openblas/lib/libopenblas.so",
        },
        "//conditions:default": {},
    }),
    lib_source = "@suitesparse//:all",
    out_shared_libs = UMFPACK_SHARED_LIBS,
    deps = ["@openblas"],
)
