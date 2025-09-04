use std::{
    env,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[cfg(any(
    all(feature = "nvcc_sm_86", feature = "nvcc_sm_80"),
    all(feature = "nvcc_sm_86", feature = "nvcc_sm_90"),
    all(feature = "nvcc_sm_80", feature = "nvcc_sm_90")
))]
compile_error!("Please select only one feature: nvcc_sm_86, nvcc_sm_80, or nvcc_sm_90.");

#[cfg(feature = "nvcc_sm_80")]
const NVCC_CONFIG: (&str, &str) = ("sm_80", "arch=compute_80,code=sm_80");
#[cfg(feature = "nvcc_sm_86")]
const NVCC_CONFIG: (&str, &str) = ("sm_86", "arch=compute_86,code=sm_86");
#[cfg(feature = "nvcc_sm_90")]
const NVCC_CONFIG: (&str, &str) = ("sm_90", "arch=compute_90,code=sm_90");

// ---------- helpers ----------
fn out_dir() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"))
}

fn run_git_clone(repo: &str, tag: &str, dest: &Path) {
    if dest.exists() {
        return;
    }
    eprintln!("[build.rs] Cloning {repo} ({tag}) -> {}", dest.display());
    if let Some(p) = dest.parent() { let _ = fs::create_dir_all(p); }
    let status = Command::new("git")
        .args([
            "clone",
            "--depth","1",
            "--branch", tag,
            repo,
            dest.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run `git` (is it installed?)");
    if !status.success() {
        panic!("git clone failed for {repo}@{tag}");
    }
}

fn cuda_root_and_includes() -> (Option<PathBuf>, Vec<PathBuf>) {
    let root = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            which::which("nvcc").ok().and_then(|p| p.parent().and_then(|bin| bin.parent()).map(|r| r.to_path_buf()))
        });
    let mut incs = Vec::new();
    if let Some(ref r) = root {
        for cand in [
            r.join("include"),
            r.join("include/libcudacxx"),
            r.join("targets/x86_64-linux/include"),
            r.join("targets/x86_64-linux/include/libcudacxx"),
        ] {
            if cand.exists() { incs.push(cand); }
        }
    }
    (root, incs)
}

fn rmm_include_dir() -> PathBuf {
    if let Ok(root) = env::var("RMM_ROOT") {
        let inc = PathBuf::from(root).join("include");
        if inc.exists() { return inc; }
    }
    let local = Path::new("thirdparty").join("rmm").join("include");
    if local.exists() { return local; }
    let out = out_dir();
    let rmm_root = out.join("rmm");
    let rmm_inc = rmm_root.join("include");
    if !rmm_inc.exists() {
        let repo = env::var("RMM_GIT").unwrap_or_else(|_| "https://github.com/rapidsai/rmm.git".to_string());
        let tag = env::var("RMM_TAG").unwrap_or_else(|_| "branch-24.06".to_string());
        run_git_clone(&repo, &tag, &rmm_root);
    }
    rmm_inc
}

fn nvtx_include_dir() -> PathBuf {
    if let Ok(root) = env::var("NVTX_ROOT") {
        let inc = PathBuf::from(&root).join("include");
        if inc.join("nvtx3/nvtx3.hpp").exists() { return inc; }
    }
    let out = out_dir();
    let nvtx_root = out.join("nvtx");
    let inc = nvtx_root.join("include");
    if !inc.join("nvtx3/nvtx3.hpp").exists() {
        let repo = env::var("NVTX_GIT").unwrap_or_else(|_| "https://github.com/NVIDIA/NVTX.git".to_string());
        let tag = env::var("NVTX_TAG").unwrap_or_else(|_| "v3.3.0-c-cpp".to_string());
        run_git_clone(&repo, &tag, &nvtx_root);
    }
    inc
}

fn cccl_include_dirs() -> Vec<PathBuf> {
    if let Ok(root) = env::var("CCCL_ROOT") {
        let inc = PathBuf::from(root).join("libcudacxx").join("include");
        if inc.exists() { return vec![inc]; }
    }
    let out = out_dir();
    let cccl_root = out.join("cccl");
    let repo = env::var("CCCL_GIT").unwrap_or_else(|_| "https://github.com/NVIDIA/cccl.git".to_string());
    let tag = env::var("CCCL_TAG").unwrap_or_else(|_| "v2.8.0".to_string());
    let libcudacxx_inc = cccl_root.join("libcudacxx").join("include");
    if !libcudacxx_inc.exists() {
        run_git_clone(&repo, &tag, &cccl_root);
    }
    let mut dirs = vec![libcudacxx_inc];
    let thrust = cccl_root.join("thrust");
    if thrust.join("thrust").exists() { dirs.push(thrust); }
    let cub = cccl_root.join("cub");
    if cub.join("cub").exists() { dirs.push(cub); }
    dirs
}

fn fmt_include_dir() -> PathBuf {
    if let Ok(root) = env::var("FMT_ROOT") {
        let inc = PathBuf::from(&root).join("include");
        if inc.join("fmt/format.h").exists() { return inc; }
    }
    let out = out_dir();
    let fmt_root = out.join("fmt");
    let inc = fmt_root.join("include");
    if !inc.join("fmt/format.h").exists() {
        let repo = env::var("FMT_GIT").unwrap_or_else(|_| "https://github.com/fmtlib/fmt.git".to_string());
        let tag = env::var("FMT_TAG").unwrap_or_else(|_| "10.2.1".to_string());
        run_git_clone(&repo, &tag, &fmt_root);
    }
    inc
}

fn spdlog_include_dir() -> PathBuf {
    if let Ok(root) = env::var("SPDLOG_ROOT") {
        let inc = PathBuf::from(&root).join("include");
        if inc.join("spdlog/spdlog.h").exists() { return inc; }
    }
    let out = out_dir();
    let spdlog_root = out.join("spdlog");
    let inc = spdlog_root.join("include");
    if !inc.join("spdlog/spdlog.h").exists() {
        let repo = env::var("SPDLOG_GIT").unwrap_or_else(|_| "https://github.com/gabime/spdlog.git".to_string());
        // v1.13.x is compatible with fmt v10
        let tag = env::var("SPDLOG_TAG").unwrap_or_else(|_| "v1.13.0".to_string());
        run_git_clone(&repo, &tag, &spdlog_root);
    }
    inc
}

fn main() {
    let curve = "FEATURE_BLS12_381";

    // account for cross-compilation [by examining environment variable]
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Set CC environment variable to choose an alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.
    let mut cc = cc::Build::new();

    let c_src_dir = PathBuf::from("src");
    let files = vec![c_src_dir.join("lib.c")];
    let mut cc_opt = None;

    match (cfg!(feature = "portable"), cfg!(feature = "force-adx")) {
        (true, false) => {
            println!("Compiling in portable mode without ISA extensions");
            cc_opt = Some("__BLST_PORTABLE__");
        }
        (false, true) => {
            if target_arch.eq("x86_64") {
                println!("Enabling ADX support via `force-adx` feature");
                cc_opt = Some("__ADX__");
            } else {
                println!("`force-adx` is ignored for non-x86_64 targets");
            }
        }
        (false, false) =>
        {
            #[cfg(target_arch = "x86_64")]
            if target_arch.eq("x86_64") && std::is_x86_feature_detected!("adx") {
                println!("Enabling ADX because it was detected on the host");
                cc_opt = Some("__ADX__");
            }
        }
        (true, true) => panic!("Cannot compile with both `portable` and `force-adx` features"),
    }

    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if !cfg!(debug_assertions) {
        cc.opt_level(2);
    }
    if let Some(def) = cc_opt {
        cc.define(def, None);
    }
    if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
        cc.include(include);
    }
    cc.files(&files).compile("blstrs_cuda");

    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        return;
    }
    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };

    let (nvcc_arch, nvcc_gencode) = NVCC_CONFIG;

    #[cfg(all(unix, not(target_os = "macos")))]
    println!("cargo:rustc-link-lib=dylib=dl");

    // Cache hints
    println!("cargo:rerun-if-env-changed=RMM_ROOT");
    println!("cargo:rerun-if-env-changed=RMM_GIT");
    println!("cargo:rerun-if-env-changed=RMM_TAG");
    println!("cargo:rerun-if-env-changed=NVTX_ROOT");
    println!("cargo:rerun-if-env-changed=NVTX_GIT");
    println!("cargo:rerun-if-env-changed=NVTX_TAG");
    println!("cargo:rerun-if-env-changed=CCCL_ROOT");
    println!("cargo:rerun-if-env-changed=CCCL_GIT");
    println!("cargo:rerun-if-env-changed=CCCL_TAG");
    println!("cargo:rerun-if-env-changed=FMT_ROOT");
    println!("cargo:rerun-if-env-changed=FMT_GIT");
    println!("cargo:rerun-if-env-changed=FMT_TAG");
    println!("cargo:rerun-if-env-changed=SPDLOG_ROOT");
    println!("cargo:rerun-if-env-changed=SPDLOG_GIT");
    println!("cargo:rerun-if-env-changed=SPDLOG_TAG");
    println!("cargo:rerun-if-changed=thirdparty/rmm");

    if nvcc.is_ok() {
        // Prepare include paths
        let rmm_inc = rmm_include_dir();
        let nvtx_inc = nvtx_include_dir();
        let cccl_incs = cccl_include_dirs();
        let fmt_inc = fmt_include_dir();
        let spdlog_inc = spdlog_include_dir();
        let (_cuda_root, cuda_incs) = cuda_root_and_includes();

        let mut nvcc = cc::Build::new();
        nvcc.cuda(true);
        nvcc.flag(&format!("-arch={}", nvcc_arch));
        nvcc.flag("-gencode").flag(nvcc_gencode);
        nvcc.flag("-lineinfo"); 
        nvcc.flag("-t0");

        // If compiling in debug mode, add the -G flag and reduce optimizations.
        if cfg!(debug_assertions) {
            println!("Compiling nvcc in debug mode: adding -G and -O0 flags");
            nvcc.flag("-lineinfo"); 
            nvcc.flag("-Xptxas").flag("-v");
            nvcc.flag("-G");
            nvcc.flag("-O0");
        } else {
            // For release mode, set the optimization level
            nvcc.flag("-O3");
            nvcc.flag("--use_fast_math");
            // Ensure RMM's logging (which pulls in fmt/spdlog) is disabled in release
            // by defining NDEBUG. This avoids requiring system fmt/spdlog headers.
            nvcc.define("NDEBUG", None);
        }

        #[cfg(not(target_env = "msvc"))]
        nvcc.flag("-Xcompiler").flag("-Wno-unused-function");
        nvcc.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);
        nvcc.define(curve, None);
        // Enable experimental memory_resource in libcudacxx for RMM
        nvcc.define("LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE", None);
        if let Some(def) = cc_opt {
            nvcc.define(def, None);
        }
        // Language/extension flags for NVCC C++ side
        nvcc.flag("-std=c++17");
        nvcc.flag("--expt-relaxed-constexpr");
        nvcc.flag("--expt-extended-lambda");

        // Include ordering: CCCL first, then fmt/spdlog, then RMM/NVTX, then CUDA
        for d in &cccl_incs {
            nvcc.include(d);
        }
        nvcc.include(&fmt_inc);
        nvcc.include(&spdlog_inc);
        nvcc.include(&rmm_inc);
        nvcc.include(&nvtx_inc);
        for d in &cuda_incs { nvcc.include(d); }
        if let Some(include) = env::var_os("DEP_BLST_C_SRC") { nvcc.include(include); }
        if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
            //nvcc.include(include);
            nvcc.include(&include); 
            let gpu_t_path = PathBuf::from(&include).join("util/gpu_t.cu");

            if gpu_t_path.exists() {
                nvcc.file(gpu_t_path);
            }
        }
        // Use header-only fmt to avoid linking libfmt
        nvcc.define("FMT_HEADER_ONLY", None);
        //nvcc.file("src/cuda/rmm_c_api.cu");
        nvcc.file("src/cuda/io_cuda.cu");
        nvcc.compile("io_cuda");

        let (cuda_root_opt, _cuda_incs2) = cuda_root_and_includes();

        #[cfg(all(unix, not(target_os = "macos")))]
        if let Some(cuda_root) = cuda_root_opt {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_root.join("lib64").display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_root.join("lib").display());
            let cupti = cuda_root.join("extras").join("CUPTI").join("lib64");
            if cupti.exists() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cupti.display());
            }
        }

        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rerun-if-changed=src/cuda/rmm_c_api.cu");
        println!("cargo:rerun-if-changed=src/cuda/io_cuda.cu");
        println!("cargo:rerun-if-env-changed=CXXFLAGS");
    }
}
