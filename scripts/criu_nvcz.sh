#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:" >&2
  echo "  $(basename "$0") dump <pid> <out.nvcz>" >&2
  echo "  $(basename "$0") dump <pid1,pid2,...> <out_dir>" >&2
  echo "  $(basename "$0") dump <pid> <s3://bucket/key|gs://bucket/key|az://container/blob>" >&2
  echo "  $(basename "$0") dump <pid1,pid2,...> <s3://bucket/prefix/|gs://bucket/prefix/|az://container/prefix/>" >&2
  echo "  $(basename "$0") restore <in.nvcz|dir|s3://...|gs://...|az://...>" >&2
  exit 1
}

[ $# -lt 1 ] && usage

CMD=${1}
shift

CUDA_CK_BIN=${CUDA_CK_BIN:-cuda-checkpoint}
CRIU_BIN=${CRIU_BIN:-criu}
NVCZ_BIN=${NVCZ_BIN:-nvcz}

command -v "${CUDA_CK_BIN}" >/dev/null 2>&1 || { echo "cuda-checkpoint not found" >&2; exit 1; }
command -v "${CRIU_BIN}" >/dev/null 2>&1 || { echo "criu not found" >&2; exit 1; }

if ! command -v "${NVCZ_BIN}" >/dev/null 2>&1 && [ ! -x "${NVCZ_BIN}" ]; then
  echo "nvcz binary not found; set NVCZ_BIN or add to PATH" >&2; exit 1
fi

# --- minimal cloud helpers (require CLIs) ---
is_cloud_uri() {
  case "$1" in
    s3://*|gs://*|az://*|azure://*) return 0;;
    *) return 1;;
  esac
}

upload_uri() {
  uri=$1
  file=$2
  case "${uri}" in
    s3://*)
      command -v aws >/dev/null 2>&1 || { echo "aws CLI not found" >&2; exit 1; }
      aws s3 cp "${file}" "${uri}" >/dev/null
      ;;
    gs://*)
      command -v gsutil >/dev/null 2>&1 || { echo "gsutil not found" >&2; exit 1; }
      gsutil cp -q "${file}" "${uri}"
      ;;
    az://*|azure://*)
      command -v az >/dev/null 2>&1 || { echo "az CLI not found" >&2; exit 1; }
      rest=${uri#az://}; rest=${rest#azure://}
      container=${rest%%/*}
      blob=${rest#*/}
      : "${AZURE_STORAGE_ACCOUNT:?AZURE_STORAGE_ACCOUNT not set}"
      az storage blob upload --account-name "${AZURE_STORAGE_ACCOUNT}" --container-name "${container}" --name "${blob}" --file "${file}" --overwrite true >/dev/null
      ;;
    *) echo "unknown uri scheme: ${uri}" >&2; exit 1;;
  esac
}

download_uri() {
  uri=$1
  file=$2
  case "${uri}" in
    s3://*)
      command -v aws >/dev/null 2>&1 || { echo "aws CLI not found" >&2; exit 1; }
      aws s3 cp "${uri}" "${file}" >/dev/null
      ;;
    gs://*)
      command -v gsutil >/dev/null 2>&1 || { echo "gsutil not found" >&2; exit 1; }
      gsutil cp -q "${uri}" "${file}"
      ;;
    az://*|azure://*)
      command -v az >/dev/null 2>&1 || { echo "az CLI not found" >&2; exit 1; }
      rest=${uri#az://}; rest=${rest#azure://}
      container=${rest%%/*}
      blob=${rest#*/}
      : "${AZURE_STORAGE_ACCOUNT:?AZURE_STORAGE_ACCOUNT not set}"
      az storage blob download --account-name "${AZURE_STORAGE_ACCOUNT}" --container-name "${container}" --name "${blob}" --file "${file}" --no-progress >/dev/null
      ;;
    *) echo "unknown uri scheme: ${uri}" >&2; exit 1;;
  esac
}

case "${CMD}" in
  dump)
    [ $# -eq 2 ] || usage
    PID_ARG=$1
    OUT=$2
    if [[ "${PID_ARG}" == *","* ]]; then
      IFS=',' read -r -a PIDS <<< "${PID_ARG}"
      if is_cloud_uri "${OUT}"; then
        case "${OUT}" in *"/") : ;; *) echo "For multi-PID to cloud, OUT must end with '/' prefix" >&2; exit 1;; esac
        for PID in "${PIDS[@]}"; do
          TMPDIR=$(mktemp -d)
          TMPFILE=$(mktemp --suffix .nvcz)
          "${CUDA_CK_BIN}" --toggle --pid "${PID}"
          "${CRIU_BIN}" dump --shell-job --images-dir "${TMPDIR}" --tree "${PID}"
          tar -C "${TMPDIR}" -cf - . | "${NVCZ_BIN}" compress --algo gdeflate --auto -o "${TMPFILE}"
          upload_uri "${OUT}${PID}.nvcz" "${TMPFILE}"
          rm -f "${TMPFILE}"
          echo "Saved: ${OUT}${PID}.nvcz"
        done
      else
        OUTDIR=${OUT}
        mkdir -p "${OUTDIR}"
        for PID in "${PIDS[@]}"; do
          TMPDIR=$(mktemp -d)
          "${CUDA_CK_BIN}" --toggle --pid "${PID}"
          "${CRIU_BIN}" dump --shell-job --images-dir "${TMPDIR}" --tree "${PID}"
          tar -C "${TMPDIR}" -cf - . | "${NVCZ_BIN}" compress --algo gdeflate --auto -o "${OUTDIR}/${PID}.nvcz"
          echo "Saved: ${OUTDIR}/${PID}.nvcz"
        done
      fi
    else
      PID=${PID_ARG}
      TMPDIR=$(mktemp -d)
      if is_cloud_uri "${OUT}"; then
        TMPFILE=$(mktemp --suffix .nvcz)
        "${CUDA_CK_BIN}" --toggle --pid "${PID}"
        "${CRIU_BIN}" dump --shell-job --images-dir "${TMPDIR}" --tree "${PID}"
        tar -C "${TMPDIR}" -cf - . | "${NVCZ_BIN}" compress --algo gdeflate --auto -o "${TMPFILE}"
        upload_uri "${OUT}" "${TMPFILE}"
        rm -f "${TMPFILE}"
        echo "Saved: ${OUT}"
      else
        "${CUDA_CK_BIN}" --toggle --pid "${PID}"
        "${CRIU_BIN}" dump --shell-job --images-dir "${TMPDIR}" --tree "${PID}"
        tar -C "${TMPDIR}" -cf - . | "${NVCZ_BIN}" compress --algo gdeflate --auto -o "${OUT}"
        echo "Saved: ${OUT}"
      fi
    fi
    ;;
  restore)
    [ $# -eq 1 ] || usage
    IN=$1
    if [ -d "${IN}" ]; then
      for F in "${IN}"/*.nvcz; do
        [ -e "$F" ] || continue
        TMPDIR=$(mktemp -d)
        "${NVCZ_BIN}" decompress -i "$F" | tar -C "${TMPDIR}" -xf -
        "${CRIU_BIN}" restore --shell-job --restore-detached --images-dir "${TMPDIR}" --pidfile "${TMPDIR}/pid"
        if [ -f "${TMPDIR}/pid" ]; then
          NEWPID=$(cat "${TMPDIR}/pid")
          "${CUDA_CK_BIN}" --toggle --pid "${NEWPID}"
          echo "Restored PID: ${NEWPID} from $(basename "$F")"
        else
          echo "Restored (pid unknown) from $(basename "$F")" >&2
        fi
      done
    elif is_cloud_uri "${IN}"; then
      TMPFILE=$(mktemp --suffix .nvcz)
      download_uri "${IN}" "${TMPFILE}"
      TMPDIR=$(mktemp -d)
      "${NVCZ_BIN}" decompress -i "${TMPFILE}" | tar -C "${TMPDIR}" -xf -
      "${CRIU_BIN}" restore --shell-job --restore-detached --images-dir "${TMPDIR}" --pidfile "${TMPDIR}/pid"
      rm -f "${TMPFILE}"
      if [ -f "${TMPDIR}/pid" ]; then
        NEWPID=$(cat "${TMPDIR}/pid")
        "${CUDA_CK_BIN}" --toggle --pid "${NEWPID}"
        echo "Restored PID: ${NEWPID}"
      else
        echo "Restored (pid unknown)" >&2
      fi
    else
      TMPDIR=$(mktemp -d)
      "${NVCZ_BIN}" decompress -i "${IN}" | tar -C "${TMPDIR}" -xf -
      "${CRIU_BIN}" restore --shell-job --restore-detached --images-dir "${TMPDIR}" --pidfile "${TMPDIR}/pid"
      if [ -f "${TMPDIR}/pid" ]; then
        NEWPID=$(cat "${TMPDIR}/pid")
        "${CUDA_CK_BIN}" --toggle --pid "${NEWPID}"
        echo "Restored PID: ${NEWPID}"
      else
        echo "Restored (pid unknown)" >&2
      fi
    fi
    ;;
  *)
    usage
    ;;
esac


