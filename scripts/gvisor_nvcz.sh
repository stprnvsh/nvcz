#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:" >&2
  echo "  $(basename "$0") dump <container_id> <out.nvcz> [--pids pid1,pid2]" >&2
  echo "  $(basename "$0") dump <id1,id2,...> <out_dir> [--pids pid1,pid2]" >&2
  echo "  $(basename "$0") dump <container_id> <s3://bucket/key|gs://bucket/key|az://container/blob> [--pids ...]" >&2
  echo "  $(basename "$0") dump <id1,id2,...> <s3://bucket/prefix/> [--pids ...]" >&2
  echo "  $(basename "$0") restore <container_id> <in.nvcz|dir|s3://...|gs://...|az://...>" >&2
  echo "Note: Uses gVisor runsc checkpoint/restore. Optionally suspend CUDA for specific PIDs pre-checkpoint." >&2
  exit 1
}

[ $# -lt 1 ] && usage

CMD=${1}
shift

CUDA_CK_BIN=${CUDA_CK_BIN:-cuda-checkpoint}
RUNSC_BIN=${RUNSC_BIN:-runsc}
NVCZ_BIN=${NVCZ_BIN:-nvcz}

command -v "${CUDA_CK_BIN}" >/dev/null 2>&1 || { echo "cuda-checkpoint not found" >&2; exit 1; }
command -v "${RUNSC_BIN}" >/dev/null 2>&1 || { echo "runsc not found" >&2; exit 1; }

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
    [ $# -ge 2 ] || usage
    IDS_ARG=$1
    OUT=$2
    PIDS_OPT=""
    if [ $# -ge 3 ]; then
      if [ "$3" = "--pids" ] && [ $# -ge 4 ]; then
        PIDS_OPT=$4
      fi
    fi

    suspend_pids() {
      [ -z "${PIDS_OPT}" ] && return 0
      IFS=',' read -r -a _P <<< "${PIDS_OPT}"
      for _pid in "${_P[@]}"; do
        "${CUDA_CK_BIN}" --toggle --pid "${_pid}"
      done
    }

    checkpoint_to_file() {
      cid=$1; outfile=$2
      TMPDIR=$(mktemp -d)
      suspend_pids
      "${RUNSC_BIN}" checkpoint --image-path "${TMPDIR}" "${cid}"
      tar -C "${TMPDIR}" -cf - . | "${NVCZ_BIN}" compress --algo gdeflate --auto -o "${outfile}"
    }

    if [[ "${IDS_ARG}" == *","* ]]; then
      IFS=',' read -r -a IDS <<< "${IDS_ARG}"
      if is_cloud_uri "${OUT}"; then
        case "${OUT}" in *"/") : ;; *) echo "For multi-ID to cloud, OUT must end with '/' prefix" >&2; exit 1;; esac
        for ID in "${IDS[@]}"; do
          TMPFILE=$(mktemp --suffix .nvcz)
          checkpoint_to_file "${ID}" "${TMPFILE}"
          upload_uri "${OUT}${ID}.nvcz" "${TMPFILE}"
          rm -f "${TMPFILE}"
          echo "Saved: ${OUT}${ID}.nvcz"
        done
      else
        OUTDIR=${OUT}
        mkdir -p "${OUTDIR}"
        for ID in "${IDS[@]}"; do
          checkpoint_to_file "${ID}" "${OUTDIR}/${ID}.nvcz"
          echo "Saved: ${OUTDIR}/${ID}.nvcz"
        done
      fi
    else
      ID=${IDS_ARG}
      if is_cloud_uri "${OUT}"; then
        TMPFILE=$(mktemp --suffix .nvcz)
        checkpoint_to_file "${ID}" "${TMPFILE}"
        upload_uri "${OUT}" "${TMPFILE}"
        rm -f "${TMPFILE}"
        echo "Saved: ${OUT}"
      else
        checkpoint_to_file "${ID}" "${OUT}"
        echo "Saved: ${OUT}"
      fi
    fi
    ;;
  restore)
    [ $# -ge 2 ] || usage
    ID=$1
    IN=$2

    resume_cuda_by_container() {
      # Attempt to get new PID from runsc state JSON
      STATE=$("${RUNSC_BIN}" state "${ID}" 2>/dev/null || true)
      NEWPID=$(echo "$STATE" | grep '"pid"' | sed -E 's/.*"pid"\s*:\s*([0-9]+).*/\1/' | head -n1)
      if [ -n "$NEWPID" ]; then
        "${CUDA_CK_BIN}" --toggle --pid "$NEWPID" || true
        echo "Restored PID: $NEWPID"
      else
        echo "Restored (pid unknown)" >&2
      fi
    }

    restore_from_dir() {
      dir=$1
      "${RUNSC_BIN}" restore --image-path "${dir}" "${ID}"
      sleep 1
      resume_cuda_by_container
    }

    if [ -d "${IN}" ]; then
      restore_from_dir "${IN}"
    elif is_cloud_uri "${IN}"; then
      TMPFILE=$(mktemp --suffix .nvcz)
      download_uri "${IN}" "${TMPFILE}"
      TMPDIR=$(mktemp -d)
      "${NVCZ_BIN}" decompress -i "${TMPFILE}" | tar -C "${TMPDIR}" -xf -
      rm -f "${TMPFILE}"
      restore_from_dir "${TMPDIR}"
    else
      TMPDIR=$(mktemp -d)
      "${NVCZ_BIN}" decompress -i "${IN}" | tar -C "${TMPDIR}" -xf -
      restore_from_dir "${TMPDIR}"
    fi
    ;;
  *)
    usage
    ;;
esac


