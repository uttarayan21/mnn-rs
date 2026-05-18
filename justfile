VERSION:= "3.5.0"

push-cachix:
    nix flake check
    nix eval .#checks.aarch64-darwin --json | jq -r '.[]' | cachix push mnn-rs
    nix eval .#checks.x86_64-linux --json | jq -r '.[]' | cachix push mnn-rs
publish:
    cargo publish --package mnn-sys
    cargo publish --package mnn

package:
    cargo package --package mnn-sys
    cargo package --package mnn

version name:
    cargo metadata --no-deps --format-version 1 | jq -r '.packages.[] | select(.name == "{{name}}") | .version'

checksums version=VERSION:
    #!/usr/bin/env bash
    set -euo pipefail
    suffixes=(
        android_armv7_armv8_cpu_opencl_vulkan
        ios_armv82_cpu_metal_coreml
        linux_x64_cpu_opencl
        windows_x64_cpu_opencl
        macos_x64_arm82_cpu_opencl_metal
    )
    auth=()
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    release_json="$(curl -sSL --fail \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "${auth[@]}" \
        "https://api.github.com/repos/alibaba/MNN/releases/tags/{{version}}")"
    jq_args=(--arg version "{{version}}")
    filter='{ version: $version, checksums: {} }'
    for s in "${suffixes[@]}"; do
        name="mnn_{{version}}_${s}.zip"
        digest="$(jq -r --arg name "$name" '.assets[] | select(.name == $name) | .digest' <<<"$release_json")"
        if [[ -z "$digest" || "$digest" == "null" ]]; then
            echo "error: no digest for $name" >&2; exit 1
        fi
        echo "  $name -> $digest" >&2
        jq_args+=(--arg "k_$s" "$s" --arg "v_$s" "$digest")
        filter+=" | .checksums[\$k_$s] = \$v_$s"
    done
    jq -n "${jq_args[@]}" "$filter" > mnn-sys/build/checksums.json
    echo "Wrote mnn-sys/build/checksums.json" >&2


download version=VERSION:
    mkdir -p downloads 
    curl -L -H "Accept: application/vnd.github+json" -H "-X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/alibaba/MNN/releases/tags/{{version}}" | jq -Sr '.assets[].browser_download_url' | xargs -n 1 curl -L -O --output-dir downloads/


