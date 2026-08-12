-- NVIDIA CUDA 编译配置

target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cuda17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cuflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cuda17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cuflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
