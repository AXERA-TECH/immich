#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 7 ]; then
  echo "usage: $0 <platform> <version> <deploy-dir> <env-file> <images-dir> <packages-dir> <output-dir>" >&2
  exit 1
fi

platform="$1"
version="$2"
deploy_dir="$3"
env_file="$4"
images_dir="$5"
packages_dir="$6"
output_dir="$7"

output_dir="$(realpath "${output_dir}")"
staging_dir="${output_dir}/bundle-${platform}"
bundle_path="${output_dir}/immich-axera-${platform}-${version}.tar.gz"

rm -rf "${staging_dir}"
mkdir -p "${staging_dir}/images" "${staging_dir}/deploy" "${staging_dir}/packages"

cp -a "${deploy_dir}/." "${staging_dir}/deploy/"
find "${staging_dir}/deploy" -maxdepth 1 -name '.env' -delete
cp "${env_file}" "${staging_dir}/deploy/example.env"
mkdir -p \
  "${staging_dir}/deploy/library" \
  "${staging_dir}/deploy/postgres" \
  "${staging_dir}/huggingface" \
  "${staging_dir}/models"

find "${images_dir}" -maxdepth 1 -type f -name '*.tar.gz' -exec cp {} "${staging_dir}/images/" \;
find "${packages_dir}" -maxdepth 1 -type f \( -name '*.whl' -o -name 'requirements.txt' \) -exec cp {} "${staging_dir}/packages/" \;

if [ -f "${deploy_dir}/README.md" ]; then
  mkdir -p "${staging_dir}/docs"
  cp "${deploy_dir}/README.md" "${staging_dir}/docs/README.md"
fi

tar -C "${staging_dir}" -czf "${bundle_path}" .
rm -rf "${staging_dir}"
