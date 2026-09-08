# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

find_package(Patch REQUIRED)

include(CPM)
CPMAddPackage(
    NAME polysolve
    GITHUB_REPOSITORY "polyfem/polysolve"
    GIT_TAG "715253d9c9fdf23469cdd6d6f6144717f9c65a51"
    PATCH_COMMAND ${Patch_EXECUTABLE} -rnN -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/polysolve_safe_spmv.patch
    COMMAND ${Patch_EXECUTABLE} -rnN -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/polysolve_skip_unused_energy.patch
)
