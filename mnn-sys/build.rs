use std::path::{Path, PathBuf};

use anyhow::*;
use build_target::Os;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[path = "build/bindgen.rs"]
mod bindgen;
#[path = "build/compile.rs"]
mod compile;
#[cfg(feature = "download")]
#[path = "build/download.rs"]
mod download;
#[path = "build/options.rs"]
mod options;

use bindgen::{mnn_c_bindgen, mnn_cpp_bindgen};
use compile::{build_cmake, mnn_c_build};
#[cfg(feature = "download")]
use download::{download_mnn_source, download_prebuilt_mnn, prebuilt_lib_link};
use options::{
    HALIDE_SEARCH, MANIFEST_DIR, MNN_COMPILE, TARGET_OS, TRACING_REPLACE, TRACING_SEARCH, VENDOR,
};

fn ensure_vendor_exists(vendor: impl AsRef<Path>) -> Result<()> {
    if vendor
        .as_ref()
        .read_dir()
        .with_context(|| format!("Vendor directory missing: {}", vendor.as_ref().display()))?
        .flatten()
        .count()
        == 0
    {
        anyhow::bail!("Vendor not found maybe you need to run \"git submodule update --init\"")
    }
    Ok(())
}

fn main() -> Result<()> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build/options.rs");
    #[cfg(feature = "download")]
    println!("cargo:rerun-if-changed=build/download.rs");
    println!("cargo:rerun-if-changed=build/bindgen.rs");
    println!("cargo:rerun-if-changed=build/compile.rs");

    let source = PathBuf::from(
        std::env::var("MNN_SRC")
            .ok()
            .unwrap_or_else(|| VENDOR.into()),
    );

    #[cfg(feature = "download")]
    {
        let version = std::env::var("MNN_VERSION").unwrap_or_else(|_| "3.6.1".to_string());
        download_prebuilt_mnn(&version, &out_dir).with_context(|| {
            format!(
                "Failed to download prebuilt MNN version {} for target {}-{}",
                version,
                *options::TARGET_ARCH,
                *TARGET_OS
            )
        })?;
        let source = download_mnn_source(&version, &out_dir).with_context(|| {
            format!(
                "Failed to download MNN source for version {} for target {}-{}",
                version,
                *options::TARGET_ARCH,
                *TARGET_OS
            )
        })?;
        mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &source)
            .with_context(|| "Failed to build mnn_c from downloaded source")?;
        mnn_c_bindgen(&source, &out_dir)
            .with_context(|| "Failed to generate mnn_c bindings from downloaded source")?;
        mnn_cpp_bindgen(&source, &out_dir)
            .with_context(|| "Failed to generate mnn_cpp bindings from downloaded source")?;
        println!("cargo:include={source}/include", source = source.display());
        prebuilt_lib_link(&out_dir)?;
        return Ok(());
    }

    ensure_vendor_exists(&source)?;
    println!("cargo:rerun-if-env-changed=MNN_SRC");
    println!("cargo:rerun-if-env-changed=MNN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_NVCC_FLAGS");
    println!("cargo:rerun-if-env-changed=CUDA_HOST_COMPILER");

    let vendor = out_dir.join("vendor");
    if !vendor.exists() {
        fs_extra::dir::copy(
            &source,
            &vendor,
            &fs_extra::dir::CopyOptions::new()
                .overwrite(true)
                .copy_inside(true),
        )
        .context("Failed to copy vendor")?;
        let intptr = vendor.join("include").join("MNN").join("HalideRuntime.h");
        #[cfg(unix)]
        std::fs::set_permissions(&intptr, std::fs::Permissions::from_mode(0o644))?;

        use itertools::Itertools;
        let intptr_contents = std::fs::read_to_string(&intptr)?;
        let patched = intptr_contents.lines().collect::<Vec<_>>();
        if let Some((idx, _)) = patched
            .iter()
            .find_position(|line| line.contains(HALIDE_SEARCH))
        {
            let patched = patched
                .into_iter()
                .enumerate()
                .filter(|(c_idx, _)| !(*c_idx == idx - 1 || (idx + 1..=idx + 3).contains(c_idx)))
                .map(|(_, c)| c)
                .collect::<Vec<_>>();

            std::fs::write(intptr, patched.join("\n"))?;
        }

        let mnn_define = vendor.join("include").join("MNN").join("MNNDefine.h");
        let patched =
            std::fs::read_to_string(&mnn_define)?.replace(TRACING_SEARCH, TRACING_REPLACE);
        #[cfg(unix)]
        std::fs::set_permissions(&mnn_define, std::fs::Permissions::from_mode(0o644))?;
        std::fs::write(mnn_define, patched)?;

        // Patch cpu_id.cc to add missing cstdint
        let cpu_id_file = vendor
            .join("source")
            .join("backend")
            .join("cpu")
            .join("x86_x64")
            .join("cpu_id.cc");
        if cpu_id_file.exists() {
            let cpu_id_contents = std::fs::read_to_string(&cpu_id_file)?;
            if !cpu_id_contents.contains("#include <cstdint>") {
                let patched = cpu_id_contents.replace(
                    "#include \"cpu_id.h\"",
                    "#include <cstdint>\n#include \"cpu_id.h\"",
                );
                #[cfg(unix)]
                std::fs::set_permissions(&cpu_id_file, std::fs::Permissions::from_mode(0o644))?;
                std::fs::write(cpu_id_file, patched)?;
            }
        }
    }

    // MNN's CUDA pool kernels launch blocks of prop.maxThreadsPerBlock (1024)
    // threads; on Blackwell (sm_120) nvcc assigns them >64 registers/thread,
    // so every launch fails with "too many resources requested" — and the
    // call site never checks the error, silently leaving garbage in the
    // output tensor. Cap the block size to the runtime's standard
    // threads_num() (128) like the rest of the backend. Applied outside the
    // copy-once block above so stale vendor copies get patched too; the
    // replacement is a no-op once applied.
    #[cfg(feature = "cuda")]
    patch_file(
        vendor.join("source/backend/cuda/execution/PoolExecution.cu"),
        "int threads_num = prop.maxThreadsPerBlock;",
        "int threads_num = (int)runtime->threads_num();",
    )?;

    if *MNN_COMPILE {
        let install_dir = out_dir.join("mnn-install");
        build_cmake(&vendor, &install_dir)?;
        println!(
            "cargo:rustc-link-search=native={}",
            install_dir.join("lib").display()
        );
        // MNN builds the CUDA backend as a shared lib (libMNN_Cuda_Main.so)
        // that never gets installed; link it straight from the cmake build
        // tree. (On Windows it's merged into MNN.lib instead.)
        #[cfg(feature = "cuda")]
        if *TARGET_OS != Os::Windows {
            println!(
                "cargo:rustc-link-search=native={}",
                out_dir.join("build/source/backend/cuda").display()
            );
        }
    } else if let core::result::Result::Ok(lib_dir) = std::env::var("MNN_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else {
        panic!("MNN_LIB_DIR not set while MNN_COMPILE is false");
    }

    mnn_c_build(PathBuf::from(MANIFEST_DIR).join("mnn_c"), &vendor)
        .with_context(|| "Failed to build mnn_c")?;
    mnn_c_bindgen(&vendor, &out_dir).with_context(|| "Failed to generate mnn_c bindings")?;
    mnn_cpp_bindgen(&vendor, &out_dir).with_context(|| "Failed to generate mnn_cpp bindings")?;
    println!("cargo:include={vendor}/include", vendor = vendor.display());

    if *TARGET_OS == Os::MacOS {
        #[cfg(feature = "metal")]
        println!("cargo:rustc-link-lib=framework=Foundation");
        #[cfg(feature = "metal")]
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        #[cfg(feature = "metal")]
        println!("cargo:rustc-link-lib=framework=Metal");
        #[cfg(feature = "coreml")]
        println!("cargo:rustc-link-lib=framework=CoreML");
        #[cfg(feature = "coreml")]
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        #[cfg(feature = "opencl")]
        println!("cargo:rustc-link-lib=framework=OpenCL");
        #[cfg(feature = "opengl")]
        println!("cargo:rustc-link-lib=framework=OpenGL");
    }
    // The CUDA backend registers itself through a static initializer
    // (source/backend/cuda/Register.cpp) that nothing references, so plain
    // archive linking drops it and CUDA sessions silently fall back to CPU;
    // whole-archive forces the registrar in. (Upstream handles this only for
    // MSVC via /WHOLEARCHIVE.)
    if cfg!(feature = "cuda") && *TARGET_OS != Os::Windows {
        println!("cargo:rustc-link-lib=static:+whole-archive=MNN");
        println!("cargo:rustc-link-lib=dylib=MNN_Cuda_Main");
    } else {
        println!("cargo:rustc-link-lib=static=MNN");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn patch_file(file: PathBuf, from: &str, to: &str) -> Result<()> {
    let contents = std::fs::read_to_string(&file)
        .with_context(|| format!("Failed to read {}", file.display()))?;
    if contents.contains(from) {
        #[cfg(unix)]
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644))?;
        std::fs::write(&file, contents.replace(from, to))?;
    }
    Ok(())
}
