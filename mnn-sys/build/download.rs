use std::path::Path;

use anyhow::*;
use sha2::Digest as _;

use crate::options::{TARGET_ARCH, TARGET_OS};

pub const SUFFIXES: [&str; 5] = [
    "android_armv7_armv8_cpu_opencl_vulkan",
    "ios_armv82_cpu_metal_coreml",
    "linux_x64_cpu_opencl",
    "windows_x64_cpu_opencl",
    "macos_x64_arm82_cpu_opencl_metal",
];

pub const CHECKSUMS: [&str; 5] = [
    "sha256:f85050dfcab114da9d389c3a4dcde8421cdce5a767aab5dbd1a5f0debc8b704a",
    "sha256:2405ef73ab406844be9d16768a82dd76bec7aefaf05634eaad2f5d7202587aa0",
    "sha256:db42a3ed0eb4af791c872afc0fc82d9a13236a834c557c679fe4c9e39209129b",
    "sha256:2243dfea8e8364beed3fccb5be17b804d89feae91cbdd4ce577f147347f07555",
    "sha256:2bb04d451fe7587107d970322cbc80083c381bc50b06dd3ae3f2349eb5c82a89",
];

const USER_AGENT: &str = concat!("mnn-rs-build/", env!("CARGO_PKG_VERSION"));

fn create_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .context("Failed to create HTTP client")
}

pub fn verify_checksum(path: impl AsRef<Path>, expected: impl AsRef<str>) -> Result<()> {
    let expected = expected.as_ref();
    if expected == "sha256:placeholder" {
        return Ok(());
    }
    let mut file = std::fs::File::open(&path).with_context(|| {
        format!(
            "Failed to open file for checksum verification: {}",
            path.as_ref().display()
        )
    })?;
    let mut hasher = sha2::Sha256::new();
    std::io::copy(&mut file, &mut hasher).with_context(|| {
        format!(
            "Failed to read file for checksum verification: {}",
            path.as_ref().display()
        )
    })?;
    let actual = format!("sha256:{:x}", hasher.finalize());
    if actual != expected {
        anyhow::bail!(
            "Checksum mismatch for {}: expected {}, got {}",
            path.as_ref().display(),
            expected,
            actual
        );
    }
    Ok(())
}

pub fn download_file(url: &str, dest_file: &Path, checksum: &str) -> Result<()> {
    if dest_file.exists() {
        eprintln!(
            "File already exists at {}, verifying checksum",
            dest_file.display()
        );
        verify_checksum(dest_file, checksum).with_context(|| {
            format!(
                "Checksum verification failed for existing file at {}, expected checksum: {}",
                dest_file.display(),
                checksum
            )
        })?;
        eprintln!("File verified, skipping download");
        return Ok(());
    }

    let client = create_client()?;
    let response = client
        .get(url)
        .send()
        .with_context(|| format!("Failed to download from {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Failed to download from {}, status: {}",
            url,
            response.status()
        );
    }

    let bytes = response
        .bytes()
        .with_context(|| format!("Failed to read response bytes from {}", url))?;

    std::fs::write(dest_file, &bytes).with_context(|| {
        format!(
            "Failed to save file from {} to {}",
            url,
            dest_file.display()
        )
    })?;

    verify_checksum(dest_file, checksum).with_context(|| {
        format!(
            "Checksum verification failed for downloaded file at {}, expected checksum: {}",
            dest_file.display(),
            checksum
        )
    })?;

    Ok(())
}

fn extract_zip(zip_path: &Path, dest: &Path, root_filter: Option<&str>) -> Result<()> {
    let file = std::fs::File::open(zip_path).with_context(|| {
        format!(
            "Failed to open zip file at {} for extraction",
            zip_path.display()
        )
    })?;

    let mut zip = zip::ZipArchive::new(file)
        .with_context(|| format!("Failed to read zip archive from {}", zip_path.display()))?;

    if let Some(root) = root_filter {
        zip.extract_unwrapped_root_dir(dest, |path| path == Path::new(root))
            .with_context(|| format!("Failed to extract archive to {}", dest.display()))?;
    } else {
        zip.extract(dest)
            .with_context(|| format!("Failed to extract archive to {}", dest.display()))?;
    }

    Ok(())
}

pub fn url_name_checksum(version: impl AsRef<str>) -> Result<(String, String, String)> {
    let version = version.as_ref();
    let pre_url =
        format!("https://github.com/alibaba/MNN/releases/download/{version}/mnn_{version}");

    let idx = match (&*TARGET_ARCH, &*TARGET_OS) {
        (build_target::Arch::AArch64 | build_target::Arch::Arm, build_target::Os::Android) => 0,
        (build_target::Arch::AArch64, build_target::Os::iOS) => 1,
        (build_target::Arch::X86_64, build_target::Os::Linux) => 2,
        (build_target::Arch::X86_64, build_target::Os::Windows) => 3,
        (build_target::Arch::X86_64 | build_target::Arch::AArch64, build_target::Os::MacOS) => 4,
        (arch, os) => anyhow::bail!("Prebuilt MNN is not available for target {}-{}", arch, os),
    };

    Ok((
        format!("{}_{}.zip", pre_url, SUFFIXES[idx]),
        format!("mnn_{version}_{}", SUFFIXES[idx]),
        CHECKSUMS[idx].to_string(),
    ))
}

pub fn download_prebuilt_mnn(version: impl AsRef<str>, out_dir: impl AsRef<Path>) -> Result<()> {
    let (url, root, checksum) = url_name_checksum(version)?;
    let dest = out_dir.as_ref().join("mnn_prebuilt");
    let dest_file = out_dir.as_ref().join("mnn_prebuilt.zip");

    download_file(&url, &dest_file, &checksum)?;
    extract_zip(&dest_file, &dest, Some(&root))?;

    Ok(())
}

pub fn download_mnn_source(
    version: impl AsRef<str>,
    out_dir: impl AsRef<Path>,
) -> Result<std::path::PathBuf> {
    let version = version.as_ref();
    let url = format!(
        "https://api.github.com/repos/alibaba/MNN/zipball/{}",
        version
    );
    let dest = out_dir.as_ref().join("mnn_source");
    let dest_file = out_dir.as_ref().join("mnn_source.zip");

    download_file(&url, &dest_file, "sha256:placeholder")?;

    if dest.exists() {
        if let Some(subdir) = dest.read_dir()?.flatten().find(|e| e.path().is_dir()) {
            return Ok(subdir.path());
        }
    }

    extract_zip(&dest_file, &dest, None)?;

    let subdir = dest
        .read_dir()?
        .flatten()
        .find(|e| e.path().is_dir())
        .map(|e| e.path())
        .with_context(|| {
            format!(
                "Failed to find extracted source directory in {}",
                dest.display()
            )
        })?;

    Ok(subdir)
}

pub fn prebuilt_lib_link(out_dir: impl AsRef<Path>) -> Result<()> {
    use build_target::Arch;

    let prebuilt_dir = out_dir.as_ref().join("mnn_prebuilt");
    let is_debug = cfg!(debug_assertions);
    let debug_string = if is_debug { "Debug" } else { "Release" };

    match (&*crate::options::TARGET_ARCH, &*TARGET_OS) {
        (Arch::AArch64 | Arch::Arm, build_target::Os::Android) => {
            let arch = if *crate::options::TARGET_ARCH == Arch::Arm {
                "armeabi-v7a"
            } else {
                "arm64-v8a"
            };
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join(arch).display()
            );
            println!("cargo:rustc-link-lib=dylib=MNN");
            println!("cargo:rustc-link-lib=dylib=MNN_Vulkan");
            println!("cargo:rustc-link-lib=dylib=MNN_CL");
            println!("cargo:rustc-link-lib=dylib=c++_shared");
            println!("cargo:rustc-link-lib=dylib=mnncore");
        }
        (Arch::AArch64, build_target::Os::iOS) => {
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join("Static").display()
            );
            println!("cargo:rustc-link-lib=dylib=MNN");
        }
        (Arch::X86_64, build_target::Os::Linux) => {
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join("lib").join(debug_string).display()
            );
            println!("cargo:rustc-link-lib=static=MNN");
        }
        (Arch::X86_64, build_target::Os::Windows) => {
            let crt = if cfg!(feature = "crt_static") {
                "MT"
            } else {
                "MD"
            };
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir
                    .join("lib")
                    .join("Release")
                    .join("static")
                    .join(crt)
                    .display()
            );
            println!("cargo:rustc-link-lib=static=MNN");
        }
        (Arch::X86_64 | Arch::AArch64, build_target::Os::MacOS) => {
            println!(
                "cargo:rustc-link-search={}",
                prebuilt_dir.join("Static").display()
            );
            println!("cargo:rustc-link-lib=MNN");
        }
        (arch, os) => anyhow::bail!("Prebuilt MNN is not available for target {}-{}", arch, os),
    };
    Ok(())
}
